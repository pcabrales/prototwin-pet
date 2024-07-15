# Range deviation
def range_loss(output, target, range=0.9):
    ''' This is the difference between output and target in the depth at which
    the dose reaches a certain percentage of the Bragg Peak dose after the Bragg Peak.
    This is done for every curve in the transversal plane where the dose is not zero.
    '''
    max_global = torch.amax(target, dim=(1, 2, 3))  # Overall max for each image
    max_along_depth, idx_max_along_depth= torch.max(target, dim=1)  # Max at each transversal point
    indices_keep = max_along_depth > 0.1 * max_global.unsqueeze(-1).unsqueeze(-1)  # Unsqueeze to match dimensions of the tensors. These are the indices of the transversal Bragg Peaks higher than 1% of the highest peak BP
    max_along_depth = max_along_depth[indices_keep] # Only keep the max, or bragg peaks, of transversal points with non-null dose
    idx_max_along_depth = idx_max_along_depth[indices_keep]
    # target_permuted = torch.permute(target, (1, 0, 2, 3))
    # output_permuted = torch.permute(output, (1, 0, 2, 3))
    target_permuted = manual_permute(target, (1, 0, 2, 3))
    output_permuted = manual_permute(output, (1, 0, 2, 3))
    new_shape = [150] + [torch.sum(indices_keep).item()]
    indices_keep = indices_keep.expand(150, -1, -1, -1)
    ddp_data = target_permuted[indices_keep].reshape(new_shape)
    ddp_output_data = output_permuted[indices_keep].reshape(new_shape)

    depth = np.arange(150)  # in mm´
    ddp = interp1d(depth, ddp_data, axis=0, kind='cubic')
    ddp_output = interp1d(depth, ddp_output_data, axis=0, kind='cubic')
    depth_extended = np.linspace(min(depth), max(depth), 10000)
    dose_at_range = range * max_along_depth.numpy()

    ddp_depth_extended = ddp(depth_extended)
    ddp_output_depth_extended = ddp_output(depth_extended)
    # n_plot = 115
    # plt.plot(depth_extended, ddp_depth_extended[:, n_plot])
    # plt.plot(depth_extended, ddp_output_depth_extended[:, n_plot])

    mask = depth_extended[:, np.newaxis] > idx_max_along_depth.numpy()  # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)
    ddp_depth_extended[mask] = 0
    ddp_output_depth_extended[mask] = 0
    depth_at_range = depth_extended[np.abs(ddp_depth_extended - dose_at_range).argmin(axis=0)]
    depth_at_range_output = depth_extended[np.abs(ddp_output_depth_extended - dose_at_range).argmin(axis=0)]

    # plt.plot(depth_at_range[n_plot], dose_at_range[n_plot], marker=".", markersize=10)
    # plt.plot(depth_at_range_output[n_plot], dose_at_range[n_plot], marker=".", markersize=10)
    return torch.tensor(depth_at_range_output - depth_at_range)
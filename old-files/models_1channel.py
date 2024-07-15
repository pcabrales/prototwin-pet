import torch as torch
import torch.nn as nn

# R2AttUNet
class upconv(nn.Module):
    def __init__(self, ch_in, ch_out, pad_left=0, pad_top=0, pad_front=0):
        super(upconv,self).__init__()
        self.upconv_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReplicationPad3d((pad_left, 0, pad_top, 0, pad_front, 0)),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.upconv_layer(x)
        return x


class recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)#nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1
      
      
class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
      
      
class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            recurrent_block(ch_out,t=t),
            recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class R2AttUNet(nn.Module):
    def __init__(self, ch_img=1, ch_output=1, t=2):  # Only one output channel
        super(R2AttUNet, self).__init__()
        
        self.pool = nn.MaxPool3d(2, stride=2)

        self.RRCNN1 = RRCNN_block(ch_in=ch_img, ch_out=32, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32, ch_out=64, t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        

        self.up5 = upconv(ch_in=512, ch_out=256, pad_top=1)
        self.att5 = attention_block(F_g=256, F_l=256, F_int=128)
        self.up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.up4 = upconv(ch_in=256, ch_out=128, pad_top=1, pad_left=1, pad_front=1)
        self.att4 = attention_block(F_g=128, F_l=128, F_int=64)
        self.up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.up3 = upconv(ch_in=128, ch_out=64, pad_left=1, pad_front=1)
        self.att3 = attention_block(F_g=64, F_l=64, F_int=32)
        self.up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.up2 = upconv(ch_in=64,ch_out=32)
        self.att2 = attention_block(F_g=32, F_l=32, F_int=16)
        self.up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv3d(32, ch_output, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Encoding path
        x1 = self.RRCNN1(x)

        x2 = self.pool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.pool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.pool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.pool(x4)
        x5 = self.RRCNN5(x5)

        # Decoding path
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_RRCNN5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4),dim=1)
        d4 = self.up_RRCNN4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3),dim=1)
        d3 = self.up_RRCNN3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2),dim=1)
        d2 = self.up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1.squeeze(1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(16, 64, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d11 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)  # Upscaling
        self.d21 = nn.Conv3d(64, 16, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e12(x_pool1))
        x_e22 = self.activation(self.e21(x_e21))
        x_pool2 = self.pool(x_e22)

        x_m1 = self.activation(self.m1(x_pool2))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e22], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e12], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_out = self.output(x_d22)

        return x_out.squeeze(1)


class UNetV2(nn.Module):
    def __init__(self):
        super(UNetV2, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling
        self.dropout = nn.Dropout3d(0.2)  # Dropout

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(16, 64, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d11 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)  # Upscaling
        self.d21 = nn.Conv3d(64, 16, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.dropout(self.activation(self.e12(x_pool1)))
        x_e22 = self.dropout(self.activation(self.e21(x_e21)))
        x_pool2 = self.pool(x_e22)

        x_m1 = self.activation(self.m1(x_pool2))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e22], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e12], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_out = self.output(x_d22)

        return x_out.squeeze(1)

class UNetV3(nn.Module):
    def __init__(self):
        super(UNetV3, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling
        self.dropout = nn.Dropout3d(0.2)  # Dropout

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.e31 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=1)  # Upscaling
        self.d11 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d21 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)  # Upscaling
        self.d31 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(16, 16, kernel_size=3, padding=1)


        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e21(x_pool1))
        x_e22 = self.activation(self.e22(x_e21))
        x_pool2 = self.pool(x_e22)

        x_e31 = self.activation(self.e31(x_pool2))
        x_e32 = self.activation(self.e32(x_e31))
        x_pool3 = self.pool(x_e32)

        x_m1 = self.activation(self.m1(x_pool3))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e32], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e22], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_upconv3 = self.upconv3(x_d22)
        x_upconv3 = torch.cat([x_upconv3, x_e12], dim=1)  # Concatenating
        x_d31 = self.activation(self.d31(x_upconv3))
        x_d32 = self.activation(self.d32(x_d31))

        x_out = self.output(x_d32)

        return x_out.squeeze(1)


class UNetV4(nn.Module):
    def __init__(self):
        super(UNetV5, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling
        self.dropout = nn.Dropout3d(0.3)  # Dropout

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.e31 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=1)  # Upscaling
        self.d11 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d21 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)  # Upscaling
        self.d31 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(16, 16, kernel_size=3, padding=1)


        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e21(x_pool1))
        x_e22 = self.dropout(self.activation(self.e22(x_e21)))
        x_pool2 = self.pool(x_e22)

        x_e31 = self.dropout(self.activation(self.e31(x_pool2)))
        x_e32 = self.dropout(self.activation(self.e32(x_e31)))
        x_pool3 = self.pool(x_e32)

        x_m1 = self.activation(self.m1(x_pool3))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e32], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e22], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_upconv3 = self.upconv3(x_d22)
        x_upconv3 = torch.cat([x_upconv3, x_e12], dim=1)  # Concatenating
        x_d31 = self.activation(self.d31(x_upconv3))
        x_d32 = self.activation(self.d32(x_d31))

        x_out = self.output(x_d32)

        return x_out.squeeze(1)


class UNetV5(nn.Module):
    def __init__(self):
        super(UNetV5, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.e31 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=1)  # Upscaling
        self.d11 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d21 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)  # Upscaling
        self.d31 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(16, 16, kernel_size=3, padding=1)


        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e21(x_pool1))
        x_e22 = self.activation(self.e22(x_e21))
        x_pool2 = self.pool(x_e22)

        x_e31 = self.activation(self.e31(x_pool2))
        x_e32 = self.activation(self.e32(x_e31))
        x_pool3 = self.pool(x_e32)

        x_m1 = self.activation(self.m1(x_pool3))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e32], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e22], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_upconv3 = self.upconv3(x_d22)
        x_upconv3 = torch.cat([x_upconv3, x_e12], dim=1)  # Concatenating
        x_d31 = self.activation(self.d31(x_upconv3))
        x_d32 = self.activation(self.d32(x_d31))

        x_out = self.output(x_d32)

        return x_out.squeeze(1)


class UNetV6(nn.Module):
    def __init__(self):
        super(UNetV6, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.e31 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.e41 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.e42 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, output_padding=(0, 1, 0))  # Upscaling
        self.d11 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=1)  # Upscaling
        self.d21 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d31 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)  # Upscaling
        self.d41 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.d42 = nn.Conv3d(16, 16, kernel_size=3, padding=1)


        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e21(x_pool1))
        x_e22 = self.activation(self.e22(x_e21))
        x_pool2 = self.pool(x_e22)

        x_e31 = self.activation(self.e31(x_pool2))
        x_e32 = self.activation(self.e32(x_e31))
        x_pool3 = self.pool(x_e32)

        x_e41 = self.activation(self.e41(x_pool3))
        x_e42 = self.activation(self.e42(x_e41))
        x_pool4 = self.pool(x_e42)

        x_m1 = self.activation(self.m1(x_pool4))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e42], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e32], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_upconv3 = self.upconv3(x_d22)
        x_upconv3 = torch.cat([x_upconv3, x_e22], dim=1)  # Concatenating
        x_d31 = self.activation(self.d31(x_upconv3))
        x_d32 = self.activation(self.d32(x_d31))

        x_upconv4 = self.upconv4(x_d32)
        x_upconv4 = torch.cat([x_upconv4, x_e12], dim=1)  # Concatenating
        x_d41 = self.activation(self.d41(x_upconv4))
        x_d42 = self.activation(self.d42(x_d41))

        x_out = self.output(x_d42)

        return x_out.squeeze(1)


class UNetV7(nn.Module):
    def __init__(self):
        super(UNetV7, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling

        # Encoder layers

        self.e11 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.e31 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.e41 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.e42 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, output_padding=(0, 1, 0))  # Upscaling
        self.d11 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, output_padding=1)  # Upscaling
        self.d21 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d31 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  # Upscaling
        self.d41 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.d42 = nn.Conv3d(32, 32, kernel_size=3, padding=1)


        self.output = nn.Conv3d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e21(x_pool1))
        x_e22 = self.activation(self.e22(x_e21))
        x_pool2 = self.pool(x_e22)

        x_e31 = self.activation(self.e31(x_pool2))
        x_e32 = self.activation(self.e32(x_e31))
        x_pool3 = self.pool(x_e32)

        x_e41 = self.activation(self.e41(x_pool3))
        x_e42 = self.activation(self.e42(x_e41))
        x_pool4 = self.pool(x_e42)

        x_m1 = self.activation(self.m1(x_pool4))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e42], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e32], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_upconv3 = self.upconv3(x_d22)
        x_upconv3 = torch.cat([x_upconv3, x_e22], dim=1)  # Concatenating
        x_d31 = self.activation(self.d31(x_upconv3))
        x_d32 = self.activation(self.d32(x_d31))

        x_upconv4 = self.upconv4(x_d32)
        x_upconv4 = torch.cat([x_upconv4, x_e12], dim=1)  # Concatenating
        x_d41 = self.activation(self.d41(x_upconv4))
        x_d42 = self.activation(self.d42(x_d41))

        x_out = self.output(x_d42)

        return x_out.squeeze(1)


class UNetV9(nn.Module):
    def __init__(self):
        super(UNetV9, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e_bn11 = nn.BatchNorm3d(16)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.e_bn12 = nn.BatchNorm3d(16)

        self.e21 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.e_bn21 = nn.BatchNorm3d(32)
        self.e22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.e_bn22 = nn.BatchNorm3d(32)

        self.e31 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.e_bn31 = nn.BatchNorm3d(64)
        self.e32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.e_bn32 = nn.BatchNorm3d(64)

        self.e41 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.e_bn41 = nn.BatchNorm3d(128)
        self.e42 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.e_bn42 = nn.BatchNorm3d(128)

        # Middle layers

        self.m1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.m_bn1 = nn.BatchNorm3d(256)
        self.m2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.m_bn2 = nn.BatchNorm3d(256)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, output_padding=(0, 1, 0))  # Upscaling
        self.d11 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.d_bn11 = nn.BatchNorm3d(128)
        self.d12 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.d_bn12 = nn.BatchNorm3d(128)

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, output_padding=1)  # Upscaling
        self.d21 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d_bn21 = nn.BatchNorm3d(64)
        self.d22 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.d_bn22 = nn.BatchNorm3d(64)

        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, output_padding=(1, 0, 1))  # Upscaling
        self.d31 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.d_bn31 = nn.BatchNorm3d(32)
        self.d32 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.d_bn32 = nn.BatchNorm3d(32)

        self.upconv4 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)  # Upscaling
        self.d41 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.d_bn41 = nn.BatchNorm3d(16)
        self.d42 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.d_bn42 = nn.BatchNorm3d(16)


        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e_bn11(self.e11(x)))
        x_e12 = self.activation(self.e_bn12(self.e12(x_e11)))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e_bn21(self.e21(x_pool1)))
        x_e22 = self.activation(self.e_bn22(self.e22(x_e21)))
        x_pool2 = self.pool(x_e22)

        x_e31 = self.activation(self.e_bn31(self.e31(x_pool2)))
        x_e32 = self.activation(self.e_bn32(self.e32(x_e31)))
        x_pool3 = self.pool(x_e32)

        x_e41 = self.activation(self.e_bn41(self.e41(x_pool3)))
        x_e42 = self.activation(self.e_bn42(self.e42(x_e41)))
        x_pool4 = self.pool(x_e42)

        x_m1 = self.activation(self.m_bn1(self.m1(x_pool4)))
        x_m2 = self.activation(self.m_bn2(self.m2(x_m1)))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e42], dim=1)  # Concatenating
        x_d11 = self.activation(self.d_bn11(self.d11(x_upconv1)))
        x_d12 = self.activation(self.d_bn12(self.d12(x_d11)))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e32], dim=1)  # Concatenating
        x_d21 = self.activation(self.d_bn21(self.d21(x_upconv2)))
        x_d22 = self.activation(self.d_bn22(self.d22(x_d21)))

        x_upconv3 = self.upconv3(x_d22)
        x_upconv3 = torch.cat([x_upconv3, x_e22], dim=1)  # Concatenating
        x_d31 = self.activation(self.d_bn31(self.d31(x_upconv3)))
        x_d32 = self.activation(self.d_bn32(self.d32(x_d31)))

        x_upconv4 = self.upconv4(x_d32)
        x_upconv4 = torch.cat([x_upconv4, x_e12], dim=1)  # Concatenating
        x_d41 = self.activation(self.d_bn41((self.d41(x_upconv4))))
        x_d42 = self.activation(self.d_bn42(self.d42(x_d41)))

        x_out = self.output(x_d42)

        return x_out.squeeze(1)




class UNetV13(nn.Module):
    def __init__(self):
        super(UNetV13, self).__init__()

        self.activation = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool3d(2, stride=2)  # Pooling

        # Encoder layers

        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.e21 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.e22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.e31 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.e32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Middle layers

        self.m1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.m2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        # Decoder layers

        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # Upscaling
        self.d11 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  # Upscaling
        self.d21 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.d22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)  # Upscaling
        self.d31 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.d32 = nn.Conv3d(16, 16, kernel_size=3, padding=1)


        self.output = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x_e11 = self.activation(self.e11(x))
        x_e12 = self.activation(self.e12(x_e11))
        x_pool1 = self.pool(x_e12)

        x_e21 = self.activation(self.e21(x_pool1))
        x_e22 = self.activation(self.e22(x_e21))
        x_pool2 = self.pool(x_e22)

        x_e31 = self.activation(self.e31(x_pool2))
        x_e32 = self.activation(self.e32(x_e31))
        x_pool3 = self.pool(x_e32)

        x_m1 = self.activation(self.m1(x_pool3))
        x_m2 = self.activation(self.m2(x_m1))

        x_upconv1 = self.upconv1(x_m2)
        x_upconv1 = torch.cat([x_upconv1, x_e32], dim=1)  # Concatenating
        x_d11 = self.activation(self.d11(x_upconv1))
        x_d12 = self.activation(self.d12(x_d11))

        x_upconv2 = self.upconv2(x_d12)
        x_upconv2 = torch.cat([x_upconv2, x_e22], dim=1)  # Concatenating
        x_d21 = self.activation(self.d21(x_upconv2))
        x_d22 = self.activation(self.d22(x_d21))

        x_upconv3 = self.upconv3(x_d22)
        x_upconv3 = torch.cat([x_upconv3, x_e12], dim=1)  # Concatenating
        x_d31 = self.activation(self.d31(x_upconv3))
        x_d32 = self.activation(self.d32(x_d31))

        x_out = self.output(x_d32)

        return x_out.squeeze(1)
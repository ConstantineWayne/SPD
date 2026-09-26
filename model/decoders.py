import torch
import torch.nn as nn



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

        
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        x = self.channel_embed(x)
        out = self.norm(x)
        return out

class fusion(nn.Module):
    def __init__(self, F_g, F_int, reduction=1, norm_layer=nn.BatchNorm2d):
        super(fusion, self).__init__()
        self.out_channels = F_int
        self.residual = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
		        nn.Conv2d(F_g, F_int//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(F_int//reduction, F_int//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=F_int//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(F_int//reduction, F_int, kernel_size=1, bias=True),
                        norm_layer(F_int) 
                        )
        self.norm = norm_layer(F_int)
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class AGM(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AGM,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            ChannelEmbed(F_int, F_int),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x*psi



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_pool_out= self.max_pool(x)
        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    



class HADer(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(HADer,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = AGM(F_g=channels[1],F_l=channels[1],F_int=channels[1])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = AGM(F_g=channels[2],F_l=channels[2],F_int=channels[2])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = AGM(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.Fusion4 = fusion(F_g=channels[0]*2,F_int=channels[0])

        self.CA3 = ChannelAttention(channels[1])
        self.Fusion3 = fusion(F_g=channels[1]*2,F_int=channels[1])

        self.CA2 = ChannelAttention(channels[2])
        self.Fusion2 = fusion(F_g=channels[2]*2,F_int=channels[2])

        self.CA1 = ChannelAttention(channels[3])
        self.Fusion1 = fusion(F_g=channels[3]*2,F_int=channels[3])
        
        self.SA = SpatialAttention(kernel_size=7)
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # DAM4
        d4_1 = self.CA4(d4)*d4
        d4_2 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(self.Fusion4(d4_1, d4_2))
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # aggregate 3
        d3 = d3 + x3
        
        # DAM3
        d3_1 = self.CA3(d3)*d3
        d3_2 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(self.Fusion3(d3_1, d3_2))
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # aggregate 2
        d2 = d2 + x2
        
        # DAM2
        d2_1 = self.CA2(d2)*d2
        d2_2 = self.SA(d2)*d2
        d2 = self.ConvBlock2(self.Fusion2(d2_1, d2_2))
        
        # upconv1
        d1 = self.Up1(d2)
        
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # aggregate 1
        d1 = d1 + x1
        
        # DAM1
        d1_1 = self.CA1(d1)*d1
        d1_2 = self.SA(d1)*d1
        d1 = self.ConvBlock1(self.Fusion1(d1_1, d1_2))
        return d4, d3, d2, d1 

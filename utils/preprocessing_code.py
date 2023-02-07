class MedianNormalize(object):
    def __init__(self, take_log=False):
        assert isinstance(take_log, bool)
        self.take_log = take_log

    def __call__(self, img, offset=0, take_log=False):
        assert isinstance(img, np.ndarray)
        assert isinstance(offset, int)
        assert img.ndim == 2 or img.ndim == 3

        orig_type = img.dtype
        imgN = img.astype(np.float64)
        if self.take_log:
            imgN = np.log1p(imgN)

        if img.ndim == 2:
            imgN = np.expand_dims(imgN, axis=0)

        for t in range(imgN.shape[0]):
            if imgN.shape[1] <= 2 * offset:
                offset_y = 0
            else:
                offset_y = offset
            y = imgN.shape[1] - 2 * offset_y

            if imgN.shape[2] <= 2 * offset:
                offset_x = 0
            else:
                offset_x = offset
            x = imgN.shape[2] - 2 * offset_x

            med = np.median(imgN[t, offset_y:offset_y+y, offset_x:offset_x+x])
            mad = np.median(np.abs(imgN[t,  offset_y:offset_y+y, offset_x:offset_x+x] - med))
            if mad == 0:
                mu = np.mean(imgN[t, offset_y:offset_y + y, offset_x:offset_x + x])
                muad = np.mean(np.abs(imgN[t, offset_y:offset_y + y, offset_x:offset_x + x] - mu))
                if muad == 0:
                    imgN[t, ...] = np.zeros_like(imgN[t, ...])
                else:
                    imgN[t, ...] = (imgN[t, ...] - med) / (1.253314 * muad)
            else:
                imgN[t, ...] = (imgN[t, ...] - med) / (1.486 * mad)

        if img.ndim == 2:
            imgN = np.squeeze(imgN, axis=0)

        if orig_type != np.float32 and orig_type != np.float64:
            print('Warning: (MedianNormalize) output type is {:s}'.format(orig_type.name))
        imgN = imgN.astype(orig_type)

        return imgN
        

def Resample2D(img, in_ori, in_spa, out_ori, out_spa, out_sz):
    img_sitk = sitk.GetImageFromArray(np.ascontiguousarray(img.astype(np.float32)))
    img_sitk.SetOrigin(in_ori)
    img_sitk.SetSpacing(in_spa)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(out_sz)
    resampler.SetOutputOrigin(out_ori)
    resampler.SetOutputSpacing(out_spa)
    resampler.SetInterpolator(sitk.sitkLinear)

    return sitk.GetArrayFromImage(resampler.Execute(img_sitk))
    
    
def predict_frame(self, frame, info):
    img = frame.astype(np.float32)
    in_min = 0
    in_max = 2 ** info['BitsStored'] - 1
    out_min = 0
    out_max = 255
    if in_max != out_max:
        img = img.astype(np.float32)
        img = (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
        img = np.rint(img)
    img.astype(np.uint8)

    # image information
    img_ori = [0, 0]
    img_spa = [info['ImageSpacing'][0], info['ImageSpacing'][1]]
    img_sz = [frame.shape[0], frame.shape[1]]

    # crop collimator
    img_edge = info['ImageEdges']
    img_c = img[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]
    img_c_ori = [img_edge[0] * img_spa[0], img_edge[2] * img_spa[1]]
    img_c_spa = img_spa
    img_c_sz = [img_c.shape[1], img_c.shape[0]]

    # resample image
    out_spa = 0.27
    img_r_sz = [int(img_c_sz[0] * img_c_spa[0] / out_spa + 0.5),
                int(img_c_sz[1] * img_c_spa[1] / out_spa + 0.5)]
    img_r_ori = [img_c_ori[0] - (img_c_spa[0] - out_spa) / 2.0,
                 img_c_ori[1] - (img_c_spa[1] - out_spa) / 2.0]
    img_r_spa = [out_spa, out_spa]
    img_r = Resample2D(img_c, img_c_ori, img_c_spa, img_r_ori, img_r_spa, img_r_sz)

    # normalize image
    normalizer = MedianNormalize()
    img_n = normalizer(img_r)
    img_n_spa = img_r_spa
    img_n_ori = img_r_ori
    img_n_sz = [img_n.shape[1], img_n.shape[0]]

    # forward
    with torch.no_grad():
        ten_in = Variable(torch.from_numpy(np.reshape(img_n, (1, 1, img_n_sz[1], img_n_sz[0]))))
        if self.gpu_flag:
            ten_in = ten_in.cuda()

        # forward
        ten_out = self.net(ten_in)
        img_out = np.squeeze(ten_out.cpu().numpy())

    # resample back
    vess = Resample2D(img_out, img_n_ori, img_n_spa, img_ori, img_spa, img_sz)


    return vess
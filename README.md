<div align="center">

<h1>CartoonDiff: Training-free Cartoon Image Generation with Diffusion Transformer Models</h1>

<div>
    <a href="" target="_blank">Feihong He</a><sup></sup>,
    <a href="" target="_blank">Gang Li</a><sup></sup>,
    <a href="" target="_blank">Lingyu Si</a><sup></sup>,
    <a href="" target="_blank">Leilei Yan</a><sup></sup>
    <a href="" target="_blank">Shimeng Hou</a><sup></sup>
    <a href="" target="_blank">Hongwei Dong</a><sup></sup>
    <a href="" target="_blank">Fanzhang Li</a><sup></sup>
</div>
<div>
    <sup></sup>School of Computer Science and Technology, Soochow University
    <sup></sup>Institute of Software, Chinese Academy of Sciences
    <sup></sup>University of Chinese Academy of Sciences
    <sup></sup>Northwestern Polytechnical University
</div>

</div>

![CartoonDiff](visuals/sample_grid_0.png)

## Setup
- We use the same environment configuration as DiT[`environment.yml`](environment.yml) file, If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.
```bash
conda env create -f environment.yml
conda activate DiT
```
- We used model <a href="https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt" target="_blank">pre-trained parameters</a> with an image resolution of 512x512.

## Sampling
![More CartoonDiff samples](visuals/sample_grid_1.png)
```bash
python sample.py --image-size 512 --seed 1
```

## CartoonDiff Code
- You can simply use the following code to replace the corresponding content in the 'diffusion/gaussian_diffusion.py' file in <a href="https://github.com/facebookresearch/DiT" target="_blank">DiT</a>.
```python
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x)
###############################Code Added###############################
        if t[0]<beta:
            sample_temp = noise.permute(0, 2, 3, 1)
            sample_temp = F.normalize(sample_temp, p=1, dim=-1)
            ones_tensor=torch.ones_like(sample_temp)
            ones_tensor[:,:,:,0]=2
            sample_temp=sample_temp+ones_tensor
            noise = sample_temp.permute(0, 3, 1, 2)
########################################################################

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise


        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```

## Parameters

- Beta: If your step is set to 100, then setting beta to around 25-30 is more appropriate.


# BibTeX
```
@article{he2023cartoondiff,
  title={Cartoondiff: Training-free Cartoon Image Generation with Diffusion Transformer Models},
  author={He, Feihong and Li, Gang and Si, Lingyu and Yan, Leilei and Hou, Shimeng and Dong, Hongwei and Li, Fanzhang},
  journal={arXiv preprint arXiv:2309.08251},
  year={2023}
}
```


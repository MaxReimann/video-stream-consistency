import re

# model id -> args for PWCNet class
MODEL_ARGS = {
    "pwcnet-norefine" : {"use_refinement" : False},
    "pwcnet-only4" : {"only_four_estimates" : True},
    "pwcnet-sref" : {"separated_refinement" : True},
    "pwcnet-sref-ft-sintel" : {"separated_refinement" : True},
    "pwcnet" : {},
    "pwcnet-ft-sintel" : {},
    "pwcnet-ft-sintel-norefine" : {"use_refinement" : False},
    "pwcnet-ft-sintel-4decoders-norefine" : {"decoder_count" : 4, "use_refinement" : False},
    "pwcnet-my-dense" : {}, # dense
    "pwcnet-my-norefine" : {"decoder_light" : False, "use_refinement" : False},
    "pwcnet-my-light" : {"decoder_light" : True},
    "pwcnet-my-lighthalfres" : {"decoder_light" : True, "halfres" : True},
    "pwcnet-my-light-norefine" : {"decoder_light" : True, "use_refinement" : False},
    "pwcnet-my-light-sepref" : {"decoder_light" : True, "separated_refinement" : True},
    "pwcnet-my-lighta75" : {"decoder_light" : True, "channel_alpha" : 0.75},
    "pwcnet-my-lighta50" : {"decoder_light" : True, "channel_alpha" : 0.5},
    "pwcnet-my-lighta25" : {"decoder_light" : True, "channel_alpha" : 0.25},
    #"pwcnet-my-light-prunetest02" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light-prunetest02.pth"},
    #"pwcnet-my-light-prunetest03" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light-prunetest03.pth"},
    #"pwcnet-my-light-prunetest04" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light-prunetest04.pth"},
    "pwcnet-my-light-prunesintel030" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light-prunesintel030.pth"},
    "pwcnet-my-light-prunesintel040" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light-prunesintel040.pth"},
    "pwcnet-my-light-prunesintel050" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light-prunesintel050.pth"},
    "pwcnet-my-seplight" : {"decoder_light" : True, "separated_decoder" : True},
    "pwcnet-my-seplightold" : {"decoder_light" : True, "separated_decoder" : True},
    "pwcnet-my-light4dec" : {"decoder_light" : True, "decoder_count" : 4},
    "pwcnet-my-light4dec1sep" : {"decoder_light" : True, "decoder_count" : 4, "separated_decoder" : "partially"},
    "pwcnet-my-light4dec-norefine" : {"decoder_light" : True, "decoder_count" : 4, "use_refinement" : False},
    "pwcnet-my-light4dec-sepref" : {"decoder_light" : True, "decoder_count" : 4, "separated_refinement" : True},
    #"pwcnet-my-light4dec-sepref-prunechairs030" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prunechairs030.pth"},
    #"pwcnet-my-light4dec-sepref-prunethings030" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prunethings030.pth"},
    #"pwcnet-my-light4dec-sepref-prunesintelshort030" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prunesintel030.pth"},
    #"pwcnet-my-light4dec-sepref-prunesintel030" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prunesintel030.pth"},
    #"pwcnet-my-light4dec-sepref-prune020" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light4dec-sepref-prunesintel020.pth"},
    "pwcnet-my-light4dec-sepref-prune030" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prunesintel030.pth"},
    "pwcnet-my-light4dec-sepref-prune040" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prunesintel040.pth"},
    "pwcnet-my-light4dec-sepref-prune050" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prunesintel050.pth"},
    "pwcnet-my-light4dec-halfref" : {"decoder_light" : True, "decoder_count" : 4},# "refinement_channel_alpha" : 0.5},
    #"pwcnet-my-light4dec-prune010" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light4dec-prune010.pth"},
    #"pwcnet-my-light4dec-pruneb010" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light4dec-pruneb010.pth"}, # only decoder/refinement
    #"pwcnet-my-light4dec-prunec010" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light4dec-prunec010.pth"}, # always prune 1 chan
    #"pwcnet-my-light4dec-pruned010" : {"custom_pth" : "pwc_pytorch/pwcnet-my-light4dec-pruned010.pth"}, # always prune 8 chans
    #"pwcnet-my-light4dec-prune020" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-prune020.pth"},
    "pwcnet-my-light4dec-prune030" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-prune030.pth"},
    "pwcnet-my-light4dec-prune040" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-prune040.pth"},
    "pwcnet-my-light4dec-prune050" : {"custom_pth" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-prune050.pth"},
    "pwcnet-my-light3dec" : {"decoder_light" : True, "decoder_count" : 3},
    "pwcnet-my-light3dec-norefine" : {"decoder_light" : True, "decoder_count" : 3, "use_refinement" : False},
    "pwcnet-my-light3dec-sepref" : {"decoder_light" : True, "decoder_count" : 3, "separated_refinement" : True},
    "pwcnet-my-light2dec" : {"decoder_light" : True, "decoder_count" : 2},
    "pwcnet-my-light2dec-norefine" : {"decoder_light" : True, "decoder_count" : 2, "use_refinement" : False},
    "pwcnet-my-light2dec-sepref" : {"decoder_light" : True, "decoder_count" : 2, "separated_refinement" : True},
    "pwcnet-my-light1dec" : {"decoder_light" : True, "decoder_count" : 1},
    "pwcnet-my-light1dec-norefine" : {"decoder_light" : True, "decoder_count" : 1, "use_refinement" : False},
    "pwcnet-my-light1dec-sepref" : {"decoder_light" : True, "decoder_count" : 1, "separated_refinement" : True},
    "pwcnet-my-light-corr3md" : {"decoder_light" : True, "decoder_corr_md" : 3},
    "pwcnet-my-light-corr2md" : {"decoder_light" : True, "decoder_corr_md" : 2},
    "pwcnet-my-part2seplight" : {"decoder_light" : True,"separated_decoder" : "partially"},
    "pwcnet-my-part2seplight-norefine" : {"decoder_light" : True, "separated_decoder" : "partially", "use_refinement" : False},
    "pwcnet-my-part2seplight-sepref" : {"decoder_light" : True, "separated_decoder" : "partially", "separated_refinement" : True},
    "pwcnet-my-part2seplight-sepref-trainedlater" : {"decoder_light" : True, "separated_decoder" : "partially", "separated_refinement" : True},
    "pwcnet-my-dense-sepref" : {"separated_refinement" : True},
    "pwcnet-my-lightcorr3md" : {"decoder_light" : True},
    "pwcnet-my-flowreg" : {"decoder_light" : False, "use_regularization" : True},
    "pwcnet-my-light-sepext" : {"extractor_separated" : True, "decoder_light" : True}
}

MODEL_FILES = {
    "pwcnet" : "pwc_pytorch/checkpoints/network-chairs-things.pth",
    "pwcnet-ft-sintel" : "pwc_pytorch/checkpoints/network-default.pth",
    "pwcnet-my-dense-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-dense-chairs-things-sintel.pth",
    "pwcnet-my-light4dec1sep-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec1sep-chairs-things-sintel.pth",
    "pwcnet-my-light4dec-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-chairs-things-sintel.pth",
    "pwcnet-my-light4dec-sepref-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-chairs-things-sintel.pth",
    "pwcnet-my-lighta50-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-lighta50-chairs-things-sintel.pth",
    "pwcnet-my-lighta75-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-lighta75-chairs-things-sintel.pth",
    "pwcnet-my-light-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-light-chairs-things-sintel.pth",
    "pwcnet-my-part2seplight-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-part2seplight-chairs-things-sintel.pth",
    "pwcnet-my-seplight-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-seplight-chairs-things-sintel.pth",
    "pwcnet-my-light4dec-sepref-prune030-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prune030-chairs-things-sintel.pth",
    "pwcnet-my-light4dec-sepref-prune040-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prune040-chairs-things-sintel.pth",
    "pwcnet-my-light4dec-sepref-prune050-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-light4dec-sepref-prune050-chairs-things-sintel.pth",


    # misc variants
    #"pwcnet" : "pwc_pytorch/network-chairs-things.pytorch",
    #"pwcnet-ft-sintel" : "pwc_pytorch/network-default.pytorch",
    #"pwcnet-ft-sintel-norefine" : "pwc_pytorch/network-default.pytorch",
    #"pwcnet-ft-sintel-4decoders-norefine" : "pwc_pytorch/network-default.pytorch",
    #"pwcnet-sref" : "pwc_pytorch/sref-chairs-things.pth",
    #"pwcnet-sref-ft-sintel" : "pwc_pytorch/sref-chairs-things-sintel.pth",
    #"pwcnet-norefine" : "pwc_pytorch/network-default.pytorch",
    #"pwcnet-only4" : "pwc_pytorch/network-default.pytorch",
    #"pwcnet-custom" : "insert-your-pwcnet-model-path-here.pth",
    #"pwcnet-my-chairs-hyper-base" : "pwc_pytorch/train/checkpoints/0130-hyper-baseline/model-024.pth", # until 30
    #"pwcnet-my-chairs-adamw" : "pwc_pytorch/train/checkpoints/0134-hyper-adamw/model-024.pth", # until 60
    #"pwcnet-my-chairs-reg-none" : "pwc_pytorch/train/checkpoints/0131-hyper-no-reg/model-024.pth", # until 24
    #"pwcnet-my-chairs-reg-manualnobias" : "pwc_pytorch/train/checkpoints/0133-hyper-manualnobiases-reg/model-024.pth", # until 24
    #"pwcnet-my-chairs-reg-manual" : "pwc_pytorch/train/checkpoints/0132-hyper-manual-reg/model-024.pth", # until 24
    #"pwcnet-my-chairs-gradientstopped" : "pwc_pytorch/train/checkpoints/0141-chairs-gradientstopped2-clean/model-024.pth", # until 24
    #"pwcnet-my-chairs" : "pwc_pytorch/train/checkpoints/0146-hyper-lossweights-uniform/model-060.pth", # until 60, done
    #"pwcnet-my-chairs-things" : "pwc_pytorch/train/checkpoints/0162-chairs-things-uniform-l1loss/model-014.pth", # until 14, done
    ##"pwcnet-my-dense-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0165-chairs-things-sintel2-uniform/model-115.pth", # until 115, done
    #"pwcnet-my-dense-chairs-things-sintel" : "pwc_pytorch/checkpoints/pwcnet-my-dense-chairs-things-sintel.pth",
    #"pwcnet-my-norefine-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0165-chairs-things-sintel2-uniform/model-115.pth", # until 115, done
    #"pwcnet-my-chairsthingsmix" : "pwc_pytorch/train/checkpoints/0138-chairsthingsmix/model-038.pth", # until 38
    #"pwcnet-my-light-chairs" : "pwc_pytorch/train/checkpoints/0163-light-chairs-uniform/model-060.pth", # until 60, done
    #"pwcnet-my-light-chairs-early" : "pwc_pytorch/train/checkpoints/0163-light-chairs-uniform/model-025.pth", # until 60, done
    #"pwcnet-my-light-chairs-things" : "pwc_pytorch/train/checkpoints/0165-light-chairs-things-uniform/model-012.pth", # until 12, done
    #"pwcnet-my-light-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0171-light-chairs-things-sintel2/model-100.pth", # until 100, done
    #"pwcnet-my-light-norefine-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0171-light-chairs-things-sintel2/model-100.pth", # until 100, done
    #"pwcnet-my-lighthalfres-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0171-light-chairs-things-sintel2/model-100.pth", # until 100, done
    #"pwcnet-my-light-chairs-things-sintel-norefine" : "pwc_pytorch/train/checkpoints/0171-light-chairs-things-sintel2/model-100.pth", # until 100, done
    #"pwcnet-my-light-sepref-chairs-things" : "pwc_pytorch/train/checkpoints/0300-light-sepref-chairs-things/model-010.pth",
    #"pwcnet-my-light-sepref-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0300-light-sepref-chairs-things-sintel2/model-100.pth",
    ## evaluating from which training stage to prune
    #"pwcnet-my-light4dec-sepref-prunechairs030-chairs-things" : "pwc_pytorch/train/checkpoints/0280-light4dec-sepref-prunechairs030-chairs-things/model-010.pth",
    #"pwcnet-my-light4dec-sepref-prunechairs030-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0280-light4dec-sepref-prunechairs030-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-light4dec-sepref-prunethings030-chairs-things" : "pwc_pytorch/train/checkpoints/0282-light4dec-sepref-prunethings030-things/model-010.pth",
    #"pwcnet-my-light4dec-sepref-prunesintelshort030-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0283-light4dec-sepref-prunesintel030-sintel2/model-100.pth",
    #"pwcnet-my-light4dec-sepref-prunesintel030-chairs-things" : "pwc_pytorch/train/checkpoints/0280-light4dec-sepref-prunechairs030-chairs-things/model-010.pth",
    #"pwcnet-my-light4dec-sepref-prunesintel030-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0290-light-sepref-prunesintel030-chairs-things-sintel2/model-100.pth",
    ## light pruned variants
    #"pwcnet-my-light-prunesintel030-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0290-light-sepref-prunesintel030-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-light-prunesintel040-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0291-light-sepref-prunesintel040-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-light-prunesintel050-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0292-light-prunesintel050-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-lighta75-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0206-light-thin-075-chairs-things-sintel2/model-120.pth",
    #"pwcnet-my-lighta50-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0207-light-thin050-chairs-things-sintel2/model-120.pth",
    #"pwcnet-my-light-corr3md-chairs" : "pwc_pytorch/train/checkpoints/0256-corr3md--chairs/model-025.pth",
    #"pwcnet-my-light-corr2md-chairs" : "pwc_pytorch/train/checkpoints/0257-corr2md--chairs/model-025.pth",
    #"pwcnet-my-seplightold-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0176-seplight-chairs-things-sintel2-uniform/model-120.pth",
    #"pwcnet-my-seplight-chairs-things" : "pwc_pytorch/train/checkpoints/0190-seplight-chairs-things-tweakconv/model-015.pth",
    #"pwcnet-my-seplight-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0194-seplight-chairs-things-sintel2-tweakconv/model-115.pth",
    #"pwcnet-my-light4dec1sep-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0276-light4dec-1sep-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-light4dec-chairs-things" : "pwc_pytorch/train/checkpoints/0184-light4dec-chairs-things/model-008.pth",
    #"pwcnet-my-light4dec-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0186-light4dec-chairs-things-sintel2/model-120.pth",
    #"pwcnet-my-light4dec-sepref-chairs" : "pwc_pytorch/train/checkpoints/0271-light4dec-sepref-chairs/model-060.pth",
    #"pwcnet-my-light4dec-sepref-chairs-things" : "pwc_pytorch/train/checkpoints/0273-light4dec-sepref-chairs-things/model-008.pth", # actually chairs-things
    #"pwcnet-my-light4dec-sepref-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0277-light4dec-sepref-chairs-sintel2/model-100.pth", # actually chairs-things-sintel2
    #"pwcnet-my-light4dec-norefine-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0277-light4dec-sepref-chairs-sintel2/model-100.pth", # actually chairs-things-sintel2
    #"pwcnet-my-light4dec-sepref-prune030-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0281-light4dec-sepref-prunesintel030-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-light4dec-sepref-prune040-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0285-light4dec-sepref-prunesintel040-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-light4dec-sepref-prune050-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0286-light4dec-sepref-prunesintel050-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-part2seplight-chairs-things" : "pwc_pytorch/train/checkpoints/0192-part2seplight-chairs-things/model-008.pth",
    #"pwcnet-my-part2seplight-chairs-things-sintel" : "pwc_pytorch/train/checkpoints/0196-part2seplight-chairs-things-sintel2/model-120.pth",
    #"pwcnet-my-part2seplight-sepref-trainedlater" : "pwc_pytorch/train/checkpoints/0215-part2seplight-sepreflater-chairsthingsmix-sintel2/model-050.pth",
    #"pwcnet-my-light-4decoders" : "pwc_pytorch/train/checkpoints/0171-light-chairs-things-sintel2/model-100.pth",
    #"pwcnet-my-light-4decoders-norefine" : "pwc_pytorch/train/checkpoints/0171-light-chairs-things-sintel2/model-100.pth",
}

def get_args(model_id):
    # eliminate dataset name
    model_id = re.sub("-chairs.*$", "", model_id)
    model_id = re.sub("-things.*$", "", model_id)
    # little magic extra: all pwcnet-my-* get legacy=False
    args = dict(MODEL_ARGS[model_id])
    if model_id.startswith("pwcnet-my"):
        args["legacy"] = False
    return args

def get_path(model_id):
    path = MODEL_FILES.get(model_id, None)
    # make sure a file is specified if there is a -chairs-something* suffix
    if len(re.findall("-chairs.*$", model_id)) and path is None:
        raise Exception("Model ID %s suggests a trained model, but not model file is specified!" % (model_id))
    return path

def get_model_ids():
    model_ids = set()
    model_ids.update(MODEL_ARGS.keys())
    model_ids.update(MODEL_FILES.keys())
    return list(model_ids)


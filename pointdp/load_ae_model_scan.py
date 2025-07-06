def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path,model_name):
    if model_name == '':
        print(f'Recovering model and checkpoint from {model_path}')
        checkpoint = torch.load(model_path)
        try:
            model.load_state_dict(checkpoint['model_state'])
        except:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state'])
            else:
                model = nn.DataParallel(model)
                model.load_state_dict(checkpoint['model_state'])
                model = model.module

        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # for backward compatibility with saved models
        if 'lr_sched_state' in checkpoint:
            lr_sched.load_state_dict(checkpoint['lr_sched_state'])
            if checkpoint['bnm_sched_state'] is not None:
                bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
        else:
            print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

        return model
    elif model_name == 'PCT':
        print('Recovering model and checkpoint from { pretrained/scanobjectnn/pct/best_checkpoint.pth},'
              'ae is pretrained & commly used')
        checkpoint = torch.load('pretrained/scanobjectnn/pct/best_checkpoint.pth')

        # pct
        net_dict = checkpoint['net']
        prefix_to_remove = 'module.'
        net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
        model.model.load_state_dict(net_dict)  # clean_pretrained

        checkpoint2 = torch.load('runs/pointmlp_ae_ScanObjectNN_diffusion/model_best_test.pth')
        net_dict2 = checkpoint2['model_state']
        prefix_to_remove = 'model.'
        state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
        prefix_to_remove = 'ae.'
        state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
        model.ae.load_state_dict(state_dict)  # pretrained ae

        return model
    elif model_name == 'Curvenet':
        print('Recovering model and checkpoint from { .pth},'
              'ae is pretrained & commly used')
        checkpoint = torch.load('pretrained/scanobjectnn/curvenet/best_checkpoint.pth')

        # curvenet
        net_dict = checkpoint['net']
        prefix_to_remove = 'module.'
        net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
        model.model.load_state_dict(net_dict)  # clean_pretrained

        checkpoint2 = torch.load('runs/pointmlp_ae_ScanObjectNN_diffusion/model_best_test.pth')
        net_dict2 = checkpoint2['model_state']
        prefix_to_remove = 'model.'
        state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
        prefix_to_remove = 'ae.'
        state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
        model.ae.load_state_dict(state_dict)  # pretrained ae

        return model
    elif model_name == 'pointnet2':
        print('Recovering model and checkpoint from { .pth},'
              'ae is pretrained & commly used')
        checkpoint = torch.load('pretrained/scanobjectnn/pointnet2/best_checkpoint.pth')

        #
        net_dict = checkpoint['net']
        prefix_to_remove = 'module.'
        net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
        model.model.load_state_dict(net_dict)  # clean_pretrained

        checkpoint2 = torch.load('runs/pointmlp_ae_ScanObjectNN_diffusion/model_best_test.pth')
        net_dict2 = checkpoint2['model_state']
        prefix_to_remove = 'model.'
        state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
        prefix_to_remove = 'ae.'
        state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
        model.ae.load_state_dict(state_dict)  # pretrained ae

        return model

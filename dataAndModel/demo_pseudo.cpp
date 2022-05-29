# Pseudo code for using the trained networks

Sketch2CADProModel gp_model;			//! grouping networks
gp_model.set_d_model_size(d_model);		//! default is 256

//! load network
std::string gp_network_fn = "/path/to/checkpoint/GP_net_config.txt";
gp_model.load_config_and_prebuild_network(gp_network_fn, 4);

//! choose the default first network version, incase there are different versions, you can set here
if (!gp_model.setup_network(0))			
{
	std::cout << "Error: cannot setup grouping network, check status!!!" << std::endl;
	exit(1);
}

// grouping input
/*
*
	inputs['inp'] tensor_info:
		dtype: DT_FLOAT
		shape: (1, -1, 256)
		name: serving_default_inp:0
	inputs['tar'] tensor_info:
		dtype: DT_FLOAT
		shape: (1, -1, 256)
		name: serving_default_tar:0
	inputs['enc_padding_mask'] tensor_info:
		dtype: DT_FLOAT
		shape: (1, 1, 1, -1)
		name: serving_default_enc_padding_mask:0
	inputs['look_ahead_mask'] tensor_info:
		dtype: DT_FLOAT
		shape: (1, 1, -1, -1)
		name: serving_default_look_ahead_mask:0
	inputs['dec_padding_mask'] tensor_info:
		dtype: DT_FLOAT
		shape: (1, 1, 1, -1)
		name: serving_default_dec_padding_mask:0
*/
std::vector<std::vector<float>> gp_net_input;  //! different input channels
gp_net_input.pusb_back(input_token);
gp_net_input.pusb_back(gp_token);
gp_net_input.pusb_back(enc_mask);
gp_net_input.pusb_back(comb_mask);
gp_net_input.pusb_back(dec_mask);

//! setup grouping network
if (!delaunay_->gp_model.set_input_tensor(gp_net_input))
{
	std::cout << "Waning: cannot fill tensor to grouper transformer, get: " << gp_net_input.size() << std::endl;
	return;
}

//! predict grouping output
std::vector<std::vector<float>> cur_gp_outputs;
if (!delaunay_->gp_model.predict_output(cur_gp_outputs))
{
	std::cout << "Error: cannot execute grouper transformer forward, please check" << std::endl;
	return;
}

//! get output: cur_gp_outputs[0]-probability, cur_gp_outputs[1]-attention
std::vector<float> gp_output_sigmoid;
sigmoid(cur_gp_outputs[0], gp_output_sigmoid);
round(gp_output_sigmoid);

//! then you can use the grouping now



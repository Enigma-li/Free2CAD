#ifndef _DEBUG
#ifdef _WITH_GPU_SUPPORT

#include "trained_model.h"


namespace Geex {

	Sketch2CADProModel::Sketch2CADProModel()
	{
		model_idx = 0;
		m_inputs.clear();
		model_dir = "";
		model_names.clear();
		input_node_names.clear();
		output_node_names.clear();
	}

	Sketch2CADProModel::~Sketch2CADProModel()
	{

	}

	void Sketch2CADProModel::set_d_model_size(int dm_size)
	{
		d_model = dm_size;
	}

	void Sketch2CADProModel::load_config_and_prebuild_network(std::string& conf_fn, int net_type)
	{
		std::ifstream in(conf_fn);
		if (!in.is_open())
		{
			std::cout << "Error: cannot open input loss file, get: " << conf_fn << std::endl;
			return;
		}

		// clear data
		model_names.clear();
		input_node_names.clear();
		output_node_names.clear();

		// set model directory
		model_dir = FileSystem::dir_name(conf_fn);
		network_type = net_type;

		// load model name
		std::string content;
		while (std::getline(in, content))
		{
			model_names.push_back(content);
		}
		in.close();

		// load per-model node configuration
		std::cout << "\n--Init: load network: " << std::endl;
		for (int mitr = 0; mitr < model_names.size(); mitr++)
		{
			std::cout << "\t" << model_names[mitr] << std::endl;
			std::string per_model_conf_fn = model_dir + "//" + model_names[mitr] + "_node_def.txt";
			std::ifstream m_model_in(per_model_conf_fn);

			if (!m_model_in.is_open())
			{
				std::cout << "Error: cannot load model " << mitr << " configuration, get: " << per_model_conf_fn << std::endl;
				continue;
			}

			std::vector<std::string> cur_inode_names;
			std::vector<int> cur_inode_cnb;
			std::vector<std::string> cur_onode_names;

			std::string content;
			while (std::getline(m_model_in, content))
			{
				std::vector<std::string> sub_strs;
				split_string(content, ' ', sub_strs);

				if (sub_strs[0].compare("Input:") == 0)
				{
					for (int nitr = 1; nitr < sub_strs.size(); nitr++)
					{
						cur_inode_names.push_back(sub_strs[nitr]);
					}
				}
				else if (sub_strs[0].compare("Output:") == 0)
				{
					for (int oitr = 1; oitr < sub_strs.size(); oitr++)
					{
						cur_onode_names.push_back(sub_strs[oitr]);
					}
				}
			}
			m_model_in.close();

			input_node_names.push_back(cur_inode_names);
			output_node_names.push_back(cur_onode_names);
		}
	}

	bool Sketch2CADProModel::setup_network(int idx)
	{
		std::cout << "\n-------Current network: " << model_names[idx] << std::endl;
		// set model index
		model_idx = idx;

		std::string saved_model_dir = model_dir + "//" + model_names[model_idx];
		tensorflow::SessionOptions session_options;
		session_options.config.mutable_gpu_options()->set_allow_growth(true);
		session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
		tensorflow::RunOptions run_options;
		tensorflow::Status m_status = LoadSavedModel(session_options, run_options, saved_model_dir, { tensorflow::kSavedModelTagServe }, &model_bundle);

		if (!m_status.ok())
		{
			std::cout << "Error: cannot load saved network, get: " << m_status.ToString() << ", with network: " << model_dir << std::endl;
			return false;
		}

		// warm up network
		if (!warmup_network())
		{
			return false;
		}

		return true;
	}

	bool Sketch2CADProModel::warmup_network()
	{
		std::cout << "\n-----Warmup network...";
		auto timer_start = std::chrono::system_clock::now();
		
		m_inputs.clear();

		if (network_type == 0)
		{
			// stroke embedding
			tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 40, iH, iW, 1 }));
			auto cur_tensor_data_ptr = cur_input.flat<float>().data();
			for (int ditr = 0; ditr < 40 * iH*iW; ditr++)
			{
				cur_tensor_data_ptr[ditr] = 0.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[0][0], cur_input));

			std::vector<std::string> output_nodes = output_node_names[model_idx];
			std::vector<tensorflow::Tensor> outputs;

			tensorflow::Status runStatus = model_bundle.GetSession()->Run(m_inputs, output_nodes, {}, &outputs);

			if (!runStatus.ok())
			{
				std::cout << "Error: cannot run predictions, get: \n" << runStatus.ToString() << std::endl;
				return false;
			}
		}
		else if (network_type == 1)
		{
			// depth embedding
			tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, iH, iW, 1 }));
			auto cur_tensor_data_ptr = cur_input.flat<float>().data();
			for (int ditr = 0; ditr < 1 * iH * iW; ditr++)
			{
				cur_tensor_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[0][0], cur_input));

			std::vector<std::string> output_nodes = output_node_names[model_idx];
			std::vector<tensorflow::Tensor> outputs;

			tensorflow::Status runStatus = model_bundle.GetSession()->Run(m_inputs, output_nodes, {}, &outputs);

			if (!runStatus.ok())
			{
				std::cout << "Error: cannot run predictions, get: \n" << runStatus.ToString() << std::endl;
				return false;
			}
		}
		else if (network_type == 2)
		{
			// normal embedding
			tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, iH, iW, 3 }));
			auto cur_tensor_data_ptr = cur_input.flat<float>().data();
			for (int ditr = 0; ditr < 1 * iH * iW * 3; ditr++)
			{
				cur_tensor_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[0][0], cur_input));

			std::vector<std::string> output_nodes = output_node_names[model_idx];
			std::vector<tensorflow::Tensor> outputs;

			tensorflow::Status runStatus = model_bundle.GetSession()->Run(m_inputs, output_nodes, {}, &outputs);

			if (!runStatus.ok())
			{
				std::cout << "Error: cannot run predictions, get: \n" << runStatus.ToString() << std::endl;
				return false;
			}
		}
		else if (network_type == 3)
		{
			// full grouping and regression network - input and forward
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
				inputs['full_stroke_input'] tensor_info:
					dtype: DT_FLOAT
					shape: (1, 256, 256, -1)
					name: serving_default_full_stroke_input:0
				inputs['group_DN_maps'] tensor_info:
					dtype: DT_FLOAT
					shape: (1, -1, 256, 256, 4)
					name: serving_default_group_DN_maps:0
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

			tensorflow::Tensor cur_inp(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 3, d_model }));
			auto cur_inp_data_ptr = cur_inp.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 3 * d_model; ditr++)
			{
				cur_inp_data_ptr[ditr] = 0.5;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][0], cur_inp));

			tensorflow::Tensor cur_tar(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, d_model }));
			auto cur_tar_data_ptr = cur_tar.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * d_model; ditr++)
			{
				cur_tar_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][1], cur_tar));

			tensorflow::Tensor cur_full_stroke(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, iW, iH, 3 }));
			auto cur_full_stroke_ptr = cur_full_stroke.flat<float>().data();
			for (int ditr = 0; ditr < 1 * iW * iH * 3; ditr++)
			{
				cur_full_stroke_ptr[ditr] = 0.5;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][2], cur_full_stroke));

			tensorflow::Tensor cur_gp_dn_map(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, iW, iH, 4 }));
			auto cur_gp_dn_ptr = cur_gp_dn_map.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * iW * iH * 4; ditr++)
			{
				cur_gp_dn_ptr[ditr] = 0.5;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][3], cur_gp_dn_map));

			tensorflow::Tensor cur_enc_mask(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, 3 }));
			auto cur_enc_mask_data_ptr = cur_enc_mask.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * 1 * 3; ditr++)
			{
				cur_enc_mask_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][4], cur_enc_mask));

			tensorflow::Tensor cur_look_mask(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, 1 }));
			auto cur_look_mask_data_ptr = cur_look_mask.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * 1 * 1; ditr++)
			{
				cur_look_mask_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][5], cur_look_mask));

			tensorflow::Tensor cur_dec_mask(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, 3 }));
			auto cur_dec_mask_data_ptr = cur_dec_mask.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * 1 * 3; ditr++)
			{
				cur_dec_mask_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][6], cur_dec_mask));

			std::vector<std::string> output_nodes = output_node_names[model_idx];
			std::vector<tensorflow::Tensor> outputs;
			tensorflow::Status runStatus = model_bundle.GetSession()->Run(m_inputs, output_nodes, {}, &outputs);

			if (!runStatus.ok())
			{
				std::cout << "Error: cannot run predictions, get: \n" << runStatus.ToString() << std::endl;
				return false;
			}
		}
		else if (network_type == 4)
		{
			// grouping network input and forward
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

			tensorflow::Tensor cur_inp(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 3, d_model }));
			auto cur_inp_data_ptr = cur_inp.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 3 * d_model; ditr++)
			{
				cur_inp_data_ptr[ditr] = 0.5;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][0], cur_inp));

			tensorflow::Tensor cur_tar(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, d_model }));
			auto cur_tar_data_ptr = cur_tar.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * d_model; ditr++)
			{
				cur_tar_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][1], cur_tar));

			tensorflow::Tensor cur_enc_mask(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, 3 }));
			auto cur_enc_mask_data_ptr = cur_enc_mask.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * 1 * 3; ditr++)
			{
				cur_enc_mask_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][2], cur_enc_mask));

			tensorflow::Tensor cur_look_mask(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, 1 }));
			auto cur_look_mask_data_ptr = cur_look_mask.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * 1 * 1; ditr++)
			{
				cur_look_mask_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][3], cur_look_mask));

			tensorflow::Tensor cur_dec_mask(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, 3 }));
			auto cur_dec_mask_data_ptr = cur_dec_mask.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 1 * 1 * 3; ditr++)
			{
				cur_dec_mask_data_ptr[ditr] = 1.0;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][4], cur_dec_mask));

			std::vector<std::string> output_nodes = output_node_names[model_idx];
			std::vector<tensorflow::Tensor> outputs;
			tensorflow::Status runStatus = model_bundle.GetSession()->Run(m_inputs, output_nodes, {}, &outputs);

			if (!runStatus.ok())
			{
				std::cout << "Error: cannot run predictions, get: \n" << runStatus.ToString() << std::endl;
				return false;
			}
		}
		else if (network_type == 5)
		{
			// Regression network 
			//	inputs['gp_ND_maps'] tensor_info:
			//		dtype: DT_FLOAT
			//		shape : (1, -1, 256, 256, 4)
			//		name : serving_default_gp_ND_maps : 0
			//	inputs['gp_S_map'] tensor_info :
			//		dtype : DT_FLOAT
			//		shape : (1, -1, 256, 256, 1)
			//		name : serving_default_gp_S_map : 0
			//	outputs['base_curve'] tensor_info :
			//		dtype : DT_FLOAT
			//		shape : (-1, -1, 256, 256, 1)
			//		name : StatefulPartitionedCall : 0
			//	outputs['face_map'] tensor_info :
			//		dtype : DT_FLOAT
			//		shape : (-1, -1, 256, 256, 1)
			//		name : StatefulPartitionedCall : 1

			tensorflow::Tensor cur_gp_stroke(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 3, iW, iH, 1 }));
			auto cur_gp_stroke_ptr = cur_gp_stroke.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 3 * iW * iH * 1; ditr++)
			{
				cur_gp_stroke_ptr[ditr] = 0.5;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][1], cur_gp_stroke));

			tensorflow::Tensor cur_gp_dn_map(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 3, iW, iH, 4 }));
			auto cur_gp_dn_ptr = cur_gp_dn_map.flat<float>().data();
			for (int ditr = 0; ditr < 1 * 3 * iW * iH * 4; ditr++)
			{
				cur_gp_dn_ptr[ditr] = 0.5;
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][0], cur_gp_dn_map));

			std::vector<std::string> output_nodes = output_node_names[model_idx];
			std::vector<tensorflow::Tensor> outputs;

			tensorflow::Status runStatus = model_bundle.GetSession()->Run(m_inputs, output_nodes, {}, &outputs);

			if (!runStatus.ok())
			{
				std::cout << "Error: cannot run predictions, get: \n" << runStatus.ToString() << std::endl;
				return false;
			}
		}

		std::cout << "done\n" << std::endl;
		auto timer_end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = timer_end - timer_start;
		std::cout << "Warmup, time: " << diff.count() << " s. \n" << std::endl;

		return true;
	}

	bool Sketch2CADProModel::set_input_tensor(std::vector<std::vector<float>>& data)
	{
		m_inputs.clear();

		if (network_type == 0)		// stroke embedding
		{
			if (input_node_names[0].size() != 1)
			{
				return false;
			}

			// fill tensor: [nb_stroke, 256, 256, 1]
			int nb_stroke = data.size();
			tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ nb_stroke, iH, iW, 1 }));
			auto cur_tensor_data_ptr = cur_input.flat<float>().data();
			for (int ditr = 0; ditr < nb_stroke; ditr++)
			{
				for (int itr = 0; itr < iH * iW; itr++)
				{
					cur_tensor_data_ptr[ditr * (iH * iW) + itr] = data[ditr][itr];
				}
			}
			m_inputs.push_back(std::make_pair(input_node_names[0][0], cur_input));
		}
		else if (network_type == 1)		// depth embedding
		{
			if (input_node_names[0].size() != 1)
			{
				return false;
			}

			int nb_gp = data.size();
			tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ nb_gp, iH, iW, 1 }));
			auto cur_tensor_data_ptr = cur_input.flat<float>().data();
			for (int ditr = 0; ditr < nb_gp; ditr++)
			{
				for (int itr = 0; itr < iH * iW; itr++)
				{
					cur_tensor_data_ptr[ditr * (iH * iW) + itr] = data[ditr][itr];
				}
			}
			m_inputs.push_back(std::make_pair(input_node_names[0][0], cur_input));
		}
		else if (network_type == 2)		// normal embedding
		{
			if (input_node_names[0].size() != 1)
			{
				return false;
			}

			int nb_gp = data.size();
			tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ nb_gp, iH, iW, 3 }));
			auto cur_tensor_data_ptr = cur_input.flat<float>().data();
			for (int ditr = 0; ditr < nb_gp; ditr++)
			{
				for (int itr = 0; itr < iH * iW * 3; itr++)
				{
					cur_tensor_data_ptr[ditr * (iH * iW * 3) + itr] = data[ditr][itr];
				}
			}
			m_inputs.push_back(std::make_pair(input_node_names[0][0], cur_input));
		}
		else if (network_type == 3)		// full grouping and regression transformer
		{
			if (data.size() != 7 || input_node_names[model_idx].size() != 7)
			{
				std::cout << "Error: not enough tensor for transformer, get: " << data.size() << std::endl;
				return false;
			}

			int nb_strokes = data[0].size() / d_model;
			int nb_group = data[1].size() / d_model;

			tensorflow::Tensor inp_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, nb_strokes, d_model }));
			auto inp_tensor_data_ptr = inp_tensor.flat<float>().data();
			for (int itr = 0; itr < data[0].size(); itr++)
			{
				inp_tensor_data_ptr[itr] = data[0][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][0], inp_tensor));
			
			tensorflow::Tensor gp_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, nb_group, d_model }));
			auto gp_tensor_data_ptr = gp_tensor.flat<float>().data();
			for (int itr = 0; itr < data[1].size(); itr++)
			{
				gp_tensor_data_ptr[itr] = data[1][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][1], gp_tensor));

			tensorflow::Tensor cur_full_stroke(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, iW, iH, nb_strokes }));
			auto cur_full_stroke_ptr = cur_full_stroke.flat<float>().data();
			for (int itr = 0; itr < 1 * iW * iH * nb_strokes; itr++)
			{
				cur_full_stroke_ptr[itr] = data[2][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][2], cur_full_stroke));

			tensorflow::Tensor cur_gp_dn_map(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, nb_group, iW, iH, 4 }));
			auto cur_gp_dn_ptr = cur_gp_dn_map.flat<float>().data();
			for (int itr = 0; itr < 1 * nb_group * iW * iH * 4; itr++)
			{
				cur_gp_dn_ptr[itr] = data[3][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][3], cur_gp_dn_map));

			tensorflow::Tensor encMask_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, nb_strokes }));
			auto encMask_tensor_data_ptr = encMask_tensor.flat<float>().data();
			for (int itr = 0; itr < data[4].size(); itr++)
			{
				encMask_tensor_data_ptr[itr] = data[4][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][4], encMask_tensor));

			tensorflow::Tensor combMask_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, nb_group, nb_group }));
			auto combMask_tensor_data_ptr = combMask_tensor.flat<float>().data();
			for (int itr = 0; itr < data[5].size(); itr++)
			{
				combMask_tensor_data_ptr[itr] = data[5][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][5], combMask_tensor));

			tensorflow::Tensor decMask_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, nb_strokes }));
			auto decMask_tensor_data_ptr = decMask_tensor.flat<float>().data();
			for (int itr = 0; itr < data[6].size(); itr++)
			{
				decMask_tensor_data_ptr[itr] = data[6][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][6], decMask_tensor));

		}
		else if (network_type == 4) // grouper transformer
		{
			if (data.size() != 5 || input_node_names[model_idx].size() != 5)
			{
				std::cout << "Error: not enough tensor for transformer, get: " << data.size() << std::endl;
				return false;
			}

			int nb_strokes = data[0].size() / d_model;
			int nb_group = data[1].size() / d_model;

			tensorflow::Tensor inp_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, nb_strokes, d_model }));
			auto inp_tensor_data_ptr = inp_tensor.flat<float>().data();
			for (int itr = 0; itr < data[0].size(); itr++)
			{
				inp_tensor_data_ptr[itr] = data[0][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][0], inp_tensor));
			
			tensorflow::Tensor gp_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, nb_group, d_model }));
			auto gp_tensor_data_ptr = gp_tensor.flat<float>().data();
			for (int itr = 0; itr < data[1].size(); itr++)
			{
				gp_tensor_data_ptr[itr] = data[1][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][1], gp_tensor));

			tensorflow::Tensor encMask_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, nb_strokes }));
			auto encMask_tensor_data_ptr = encMask_tensor.flat<float>().data();
			for (int itr = 0; itr < data[2].size(); itr++)
			{
				encMask_tensor_data_ptr[itr] = data[2][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][2], encMask_tensor));

			tensorflow::Tensor combMask_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, nb_group, nb_group }));
			auto combMask_tensor_data_ptr = combMask_tensor.flat<float>().data();
			for (int itr = 0; itr < data[3].size(); itr++)
			{
				combMask_tensor_data_ptr[itr] = data[3][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][3], combMask_tensor));

			tensorflow::Tensor decMask_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 1, 1, nb_strokes }));
			auto decMask_tensor_data_ptr = decMask_tensor.flat<float>().data();
			for (int itr = 0; itr < data[4].size(); itr++)
			{
				decMask_tensor_data_ptr[itr] = data[4][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][4], decMask_tensor));
		}
		else if (network_type == 5) // regressor
		{
			int nb_group = data[1].size() / iH / iW;

			if (data.size() != 2 || input_node_names[model_idx].size() != 2)
			{
				std::cout << "Error: not enough tensor for transformer, get: " << data.size() << std::endl;
				return false;
			}

			tensorflow::Tensor cur_gp_dn_map(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, nb_group, iW, iH, 4 }));
			auto cur_gp_dn_ptr = cur_gp_dn_map.flat<float>().data();
			for (int itr = 0; itr < 1 * nb_group * iW * iH * 4; itr++)
			{
				cur_gp_dn_ptr[itr] = data[0][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][0], cur_gp_dn_map));

			tensorflow::Tensor cur_gp_stroke(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, nb_group, iW, iH, 1 }));
			auto cur_gp_stroke_ptr = cur_gp_stroke.flat<float>().data();
			for (int itr = 0; itr < 1 * nb_group * iW * iH * 1; itr++)
			{
				cur_gp_stroke_ptr[itr] = data[1][itr];
			}
			m_inputs.push_back(std::make_pair(input_node_names[model_idx][1], cur_gp_stroke));
		}

		return true;
	}

	bool Sketch2CADProModel::predict_output(std::vector<std::vector<float>>& net_outputs)
	{
		net_outputs.clear();
		std::vector<std::string> output_nodes = output_node_names[model_idx];
		std::vector<tensorflow::Tensor> outputs;
		tensorflow::Status runStatus = model_bundle.GetSession()->Run(m_inputs, output_nodes, {}, &outputs);

		if (!runStatus.ok())
		{
			std::cout << "Error: cannot run predictions, get: \n" << runStatus.ToString() << std::endl;
			return false;
		}

		// get output tensor
		for (int opt_itr = 0; opt_itr < output_nodes.size(); opt_itr++)
		{
			tensorflow::Tensor cur_t = outputs[opt_itr];
			std::vector<float> cur_net_out;

			int nb_elts = cur_t.NumElements();

			// debug
			//std::cout << "Current tensor elements: " << nb_elts << ", dims: " << cur_t.dims() << std::endl;

			auto tensor_data = cur_t.flat<float>().data();
			for (int d_itr = 0; d_itr < nb_elts; d_itr++)
			{
				cur_net_out.push_back(tensor_data[d_itr]);
			}

			net_outputs.push_back(cur_net_out);
		}

		return true;
	}

}

#endif
#endif
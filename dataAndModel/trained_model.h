#ifndef _DEBUG
#ifdef _WITH_GPU_SUPPORT

#pragma once


#include <vector>
#include <string>
#include <ostream>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <utils/basics/file_system.h>
#include "util_functions.h"
#include <limits>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"


namespace Geex {

	class Sketch2CADProModel {

	public:
		Sketch2CADProModel();
		~Sketch2CADProModel();

		// set d_model
		void set_d_model_size(int dm_size);

		// load config and preprocessing
		void load_config_and_prebuild_network(std::string& conf_fn, int net_type);

		// set up network
		bool setup_network(int idx);
		int get_network_idx() { return model_idx; }

		// warm up network
		bool warmup_network();

		// set input tensor
		bool set_input_tensor(std::vector<std::vector<float>>& data);

		// predict output
		bool predict_output(std::vector<std::vector<float>>& net_outputs);

	private:
		tensorflow::SavedModelBundleLite model_bundle;							//! model bundle
		std::vector<std::pair<std::string, tensorflow::Tensor>> m_inputs;		//! input name tensor pairs

		int model_idx;												//! current model index
		std::string model_dir;										//! model folder
		std::vector<std::string> model_names;						//! model names
		std::vector<std::vector<std::string>> input_node_names;		//! input node names
		std::vector<std::vector<std::string>> output_node_names;	//! output node names
		const int iH = 256, iW = 256;								//! image size
		int d_model = 256;											//! code size or embedding size
		int network_type = -1;										//! 0-stroke embedding; 1-depth embedding; 2-normal embedding; 3-transformer

	};

}

#endif
#endif
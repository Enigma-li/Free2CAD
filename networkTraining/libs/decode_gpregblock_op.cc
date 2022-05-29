//
// Project Free2CAD
//
//   Author: Changjian Li (chjili2011@gmail.com),
//   Copyright (c) 2022. All Rights Reserved.
//
//==============================================================================

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "zlib.h"

#include <complex>
#include <vector>
#include <iostream>

namespace tensorflow {

    REGISTER_OP("DecodeGpregblock")
            .Input("byte_stream: string")
            .Attr("tensor_size: list(int) >= 2")
            .Output("input_data: float")
            .Output("glabel_data: float")
            .Output("gmap_data: float")
            .Output("gdir_data: float")
            .Output("gdis_data: float")
            .Output("gbstype_data: int32")
            .Output("gbltype_data: int32")
            .Output("gnd_data: float")
            .Output("gbseg_data: float")
            .Output("presmap_data: float")
            .Output("start_index: int32")
            .Output("gfacemap_data: float")
            .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
                std::vector<int32> t_size;
                TF_RETURN_IF_ERROR(c->GetAttr("tensor_size", &t_size));
                c->set_output(0, c->MakeShape({t_size[0], t_size[1], c->UnknownDim()}));
                c->set_output(1, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
                c->set_output(2, c->MakeShape({c->UnknownDim(), t_size[0], t_size[1], 5}));
                c->set_output(3, c->MakeShape({c->UnknownDim(), 3}));
                c->set_output(4, c->MakeShape({c->UnknownDim()}));
                c->set_output(5, c->MakeShape({c->UnknownDim()}));
                c->set_output(6, c->MakeShape({c->UnknownDim()}));
                c->set_output(7, c->MakeShape({t_size[0], t_size[1], 4}));
                c->set_output(8, c->MakeShape({c->UnknownDim(), t_size[0], t_size[1], 1}));
                c->set_output(9, c->MakeShape({t_size[0], t_size[1]}));
                c->set_output(10, c->MakeShape({1}));
                c->set_output(11, c->MakeShape({c->UnknownDim(), t_size[0], t_size[1], 1}));
                return Status::OK();
            })
            .Doc(R"doc(The decoder of transformer regression image data block)doc");

    class DecodeGpregblockOp : public OpKernel {
    public:
        explicit DecodeGpregblockOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &this->tensor_size));
            OP_REQUIRES(context, this->tensor_size.size() == 2, errors::InvalidArgument("target tensor size must be 2-d, got ", this->tensor_size.size()));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& contents = context->input(0);
            OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()), errors::InvalidArgument("DecodeGpregkblock expect a scalar, got shape ",
                                                                                                       contents.shape().DebugString()));
            const StringPiece input_bytes = contents.scalar<tensorflow::tstring>()();
            // uncompress the byte stream
            int out_data_size = -1;
            float* inflate_data = inflation_byte(input_bytes, out_data_size);
            OP_REQUIRES(context, out_data_size > 0, errors::InvalidArgument("Zlib inflation error, got size: ", out_data_size));
            int64 nb_channel = (int64)inflate_data[0];
            OP_REQUIRES(context, (out_data_size - (int64)(this->tensor_size[0]*this->tensor_size[1]*nb_channel)) == 0, errors::InvalidArgument("Inflated data mismatch, got ", out_data_size));

            // get dimension
            int64 height = this->tensor_size[0];
            int64 width = this->tensor_size[1];
            int64 nb_groups = (int64)inflate_data[nb_channel];
            int64 nb_strokes = (int64)inflate_data[2*nb_channel];
            int start_idx = (int)inflate_data[3*nb_channel];

            // allocate the output tensor
            Tensor* input_data_tensor = nullptr;
            std::vector<int64> input_tensor_size;
            input_tensor_size.push_back(height);
            input_tensor_size.push_back(width);
            input_tensor_size.push_back(nb_strokes);
            TensorShape input_tensor_shape = TensorShape(gtl::ArraySlice<int64>{input_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("input_data", input_tensor_shape, &input_data_tensor));

            Tensor* glabel_data_tensor = nullptr;
            std::vector<int64> glabel_tensor_size;
            glabel_tensor_size.push_back(nb_groups);
            glabel_tensor_size.push_back(nb_strokes);
            TensorShape glabel_tensor_shape = TensorShape(gtl::ArraySlice<int64>{glabel_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("glabel_data", glabel_tensor_shape, &glabel_data_tensor));

            Tensor* gmap_data_tensor = nullptr;
            std::vector<int64> gmap_tensor_size;
            gmap_tensor_size.push_back(nb_groups);
            gmap_tensor_size.push_back(height);
            gmap_tensor_size.push_back(width);
            gmap_tensor_size.push_back(5);
            TensorShape gmap_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gmap_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gmap_data", gmap_tensor_shape, &gmap_data_tensor));

            Tensor* gdir_data_tensor = nullptr;
            std::vector<int64> gdir_tensor_size;
            gdir_tensor_size.push_back(nb_groups);
            gdir_tensor_size.push_back(3);
            TensorShape gdir_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gdir_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gdir_data", gdir_tensor_shape, &gdir_data_tensor));

            Tensor* gdis_data_tensor = nullptr;
            std::vector<int64> gdis_tensor_size;
            gdis_tensor_size.push_back(nb_groups);
            TensorShape gdis_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gdis_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gdis_data", gdis_tensor_shape, &gdis_data_tensor));

            Tensor* gbstype_data_tensor = nullptr;
            std::vector<int64> gbstype_tensor_size;
            gbstype_tensor_size.push_back(nb_groups);
            TensorShape gbstype_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gbstype_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gbstype_data", gbstype_tensor_shape, &gbstype_data_tensor));

            Tensor* gbltype_data_tensor = nullptr;
            std::vector<int64> gbltype_tensor_size;
            gbltype_tensor_size.push_back(nb_groups);
            TensorShape gbltype_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gbltype_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gbltype_data", gbltype_tensor_shape, &gbltype_data_tensor));

            Tensor* gnd_data_tensor = nullptr;
            std::vector<int64> gnd_tensor_size;
            gnd_tensor_size.push_back(height);
            gnd_tensor_size.push_back(width);
            gnd_tensor_size.push_back(4);
            TensorShape gnd_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gnd_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gnd_data", gnd_tensor_shape, &gnd_data_tensor));

            Tensor* gbseg_data_tensor = nullptr;
            std::vector<int64> gbseg_tensor_size;
            gbseg_tensor_size.push_back(nb_groups);
            gbseg_tensor_size.push_back(height);
            gbseg_tensor_size.push_back(width);
            gbseg_tensor_size.push_back(1);
            TensorShape gbseg_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gbseg_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gbseg_data", gbseg_tensor_shape, &gbseg_data_tensor));

            Tensor* pregp_data_tensor = nullptr;
            std::vector<int64> pregp_tensor_size;
            pregp_tensor_size.push_back(height);
            pregp_tensor_size.push_back(width);
            TensorShape pregp_tensor_shape = TensorShape(gtl::ArraySlice<int64>{pregp_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("presmap_data", pregp_tensor_shape, &pregp_data_tensor));

            Tensor* startidx_data_tensor = nullptr;
            std::vector<int64> startidx_tensor_size;
            startidx_tensor_size.push_back(1);
            TensorShape startidx_tensor_shape = TensorShape(gtl::ArraySlice<int64>{startidx_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("start_index", startidx_tensor_shape, &startidx_data_tensor));

            Tensor* gfacemap_data_tensor = nullptr;
            std::vector<int64> gfacemap_tensor_size;
            gfacemap_tensor_size.push_back(nb_groups);
            gfacemap_tensor_size.push_back(height);
            gfacemap_tensor_size.push_back(width);
            gfacemap_tensor_size.push_back(1);
            TensorShape gfacemap_tensor_shape = TensorShape(gtl::ArraySlice<int64>{gfacemap_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("gfacemap_data", gfacemap_tensor_shape, &gfacemap_data_tensor));

            // Assemble data into tensor
            // strokes
            auto input_data_ptr = input_data_tensor->flat<float>();
            for (int sitr = 0; sitr < nb_strokes; sitr++)
            {
                for(int ritr=0; ritr<height; ritr++)
                {
                    for(int citr=0; citr<width; citr++)
                    {
                        int64 idx = ritr*width + citr;

                        input_data_ptr(idx*nb_strokes+sitr) =inflate_data[idx*nb_channel+1+sitr];
                    }
                }
            }

            // global normal, depth data
            auto gnd_data_ptr = gnd_data_tensor->flat<float>();
            for(int ritr=0; ritr<height; ritr++)
            {
                for(int citr=0; citr<width; citr++)
                {
                    int64 idx = ritr*width + citr;

                    gnd_data_ptr(idx*4 + 0) = inflate_data[idx*nb_channel + 1 + nb_strokes + 0];
                    gnd_data_ptr(idx*4 + 1) = inflate_data[idx*nb_channel + 1 + nb_strokes + 1];
                    gnd_data_ptr(idx*4 + 2) = inflate_data[idx*nb_channel + 1 + nb_strokes + 2];
                    gnd_data_ptr(idx*4 + 3) = inflate_data[idx*nb_channel + 1 + nb_strokes + 3];
                }
            }

            // pregp image
            auto pregp_data_ptr = pregp_data_tensor->flat<float>();
            for(int ritr=0; ritr<height; ritr++)
            {
                for(int citr=0; citr<width; citr++)
                {
                    int idx = ritr*width + citr;

                    pregp_data_ptr(idx) =inflate_data[idx*nb_channel + 1 + nb_strokes + 1 + 3 + nb_groups*9];
                }
            }

            auto startidx_data_ptr = startidx_data_tensor->flat<int>();
            startidx_data_ptr(0) = start_idx;

            // group data
            auto glabel_data_ptr = glabel_data_tensor->flat<float>();
            auto gmap_data_ptr = gmap_data_tensor->flat<float>();
            auto gdir_data_ptr = gdir_data_tensor->flat<float>();
            auto gdis_data_ptr = gdis_data_tensor->flat<float>();
            auto gbstype_data_ptr = gbstype_data_tensor->flat<int>();
            auto gbltype_data_ptr = gbltype_data_tensor->flat<int>();
            auto gbseg_data_ptr = gbseg_data_tensor->flat<float>();
            auto gfacemap_data_ptr = gfacemap_data_tensor->flat<float>();    

            for(int gitr=0; gitr<nb_groups; gitr++)
            {
                int outer_idx = gitr*(width*height*5);
                for(int ritr=0; ritr<height; ritr++)
                {
                    for(int citr=0; citr<width; citr++)
                    {
                        int cur_idx = ritr*width + citr;

                        // group label
                        if(ritr==0 && citr<nb_strokes)
                        {
                            glabel_data_ptr(gitr*nb_strokes+cur_idx) =
                            inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+5];
                        }

                        // corner map, depth map, normal map
                        gmap_data_ptr(outer_idx + cur_idx*5+0) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+0];
                        gmap_data_ptr(outer_idx + cur_idx*5+1) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+1];
                        gmap_data_ptr(outer_idx + cur_idx*5+2) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+2];
                        gmap_data_ptr(outer_idx + cur_idx*5+3) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+3];
                        gmap_data_ptr(outer_idx + cur_idx*5+4) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+4];

                        // base line segmentation
                        gbseg_data_ptr(gitr*width*height + cur_idx) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+7];
                        gfacemap_data_ptr(gitr*width*height + cur_idx) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+8];

                        // direction
                        if(ritr==0 && citr<3)
                        {
                            gdir_data_ptr(gitr*3+citr) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+6];
                        }

                        // distance
                        if(ritr==0 && citr==6)
                        {
                            gdis_data_ptr(gitr) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+6];
                        }

                        // boolean type
                        if(ritr==0 && citr==7)
                        {
                            gbltype_data_ptr(gitr) = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+6];
                        }

                        // base shape type
                        if(ritr==0 && citr==8)
                        {
                            int bs_type = inflate_data[cur_idx*nb_channel+1+nb_strokes+4+gitr*9+6];
                            if(bs_type == 0 || bs_type == 1)
                                gbstype_data_ptr(gitr) = 0;
                            else
                                gbstype_data_ptr(gitr) = 1;
                        }
                    }
                }
            }

            delete[] inflate_data;
        }

    private:
        float* inflation_byte(const StringPiece &input_bytes, int& out_size)
        {
            // zipper stream
            z_stream infstream;
            infstream.zalloc = Z_NULL;
            infstream.zfree = Z_NULL;
            infstream.opaque = Z_NULL;

            // set input, output
            Byte* uncompressed_data = new Byte[100000000];
            //  delete it outside

            infstream.avail_in = (uInt)input_bytes.size();
            infstream.next_in = (Bytef*)input_bytes.data();
            infstream.avail_out = (uLong)100000000;
            infstream.next_out = uncompressed_data;

            // uncompress work
            int64 nErr, real_out_size = -1;

            nErr = inflateInit(&infstream);
            if(nErr != Z_OK)
            {
                out_size = -1;
                return nullptr;
            }
            nErr = inflate(&infstream, Z_FINISH);
            if(nErr == Z_STREAM_END)
            {
                real_out_size = (int64)infstream.total_out;
            }
            inflateEnd(&infstream);

            // assign data
            real_out_size /= 4;
            out_size = real_out_size;

            return (float *)uncompressed_data;
        }

    private:
        std::vector<int64> tensor_size;
    };

    REGISTER_KERNEL_BUILDER(Name("DecodeGpregblock").Device(DEVICE_CPU), DecodeGpregblockOp);

}

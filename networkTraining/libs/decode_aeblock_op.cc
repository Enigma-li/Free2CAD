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

    REGISTER_OP("DecodeAeblock")
            .Input("byte_stream: string")
            .Attr("tensor_size: list(int) >= 2")
            .Output("input_data: float")
            .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
                std::vector<int32> t_size;
                TF_RETURN_IF_ERROR(c->GetAttr("tensor_size", &t_size));
                c->set_output(0, c->MakeShape({t_size[0], t_size[1]}));
                return Status::OK();
            })
            .Doc(R"doc(The decoder of auto-encoder image data block)doc");

    class DecodeAeblockOp : public OpKernel {
    public:
        explicit DecodeAeblockOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &this->tensor_size));
            OP_REQUIRES(context, this->tensor_size.size() == 2, errors::InvalidArgument("target tensor size must be 2-d, got ", this->tensor_size.size()));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& contents = context->input(0);
            OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()), errors::InvalidArgument("DecodeAeblock expect a scalar, got shape ",
                                                                                                       contents.shape().DebugString()));
            const StringPiece input_bytes = contents.scalar<tensorflow::tstring>()();

            // allocate the output tensor
            Tensor* input_data_tensor = nullptr;
            std::vector<int64> input_tensor_size;
            input_tensor_size.push_back(this->tensor_size[0]);
            input_tensor_size.push_back(this->tensor_size[1]);
            TensorShape input_tensor_shape = TensorShape(gtl::ArraySlice<int64>{input_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("input_data", input_tensor_shape, &input_data_tensor));

            // assemble data into tensor
            auto input_data_ptr = input_data_tensor->flat<float>();

            // uncompress the byte stream
            int out_data_size = -1;
            float* inflate_data = inflation_byte(input_bytes, out_data_size);
            OP_REQUIRES(context, out_data_size > 0, errors::InvalidArgument("Zlib inflation error, got size: ", out_data_size));
            OP_REQUIRES(context, (out_data_size - (int)(this->tensor_size[0]*this->tensor_size[1])) == 0, errors::InvalidArgument("Inflated data mismatch, got ", out_data_size));

            // set tensor value
            int64 height = this->tensor_size[0];
            int64 width = this->tensor_size[1];

            for(int itr=0; itr<height*height; itr++)
            {
                input_data_ptr(itr) = inflate_data[itr];  // user stroke
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
            int nErr, real_out_size = -1;

            nErr = inflateInit(&infstream);
            if(nErr != Z_OK)
            {
                out_size = -1;
                return nullptr;
            }
            nErr = inflate(&infstream, Z_FINISH);
            if(nErr == Z_STREAM_END)
            {
                real_out_size = (int)infstream.total_out;
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

    REGISTER_KERNEL_BUILDER(Name("DecodeAeblock").Device(DEVICE_CPU), DecodeAeblockOp);

}

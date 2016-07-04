#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {


template <typename Dtype>
void BoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    //CHECK_EQ(bottom.size(), 1) << "Box Layer takes a single blob as input.";
    //CHECK_GE(top.size(), 1)   << "Box Layer takes exactly multiple blobs as output.";

    LOG(INFO)<< "top.size is  " <<top.size();
    count_ = bottom[0]->count();
    /* clipping rect */
    //xcoord_ = this->layer_param_.box_param().xcoord();
    //ycoord_ = this->layer_param_.box_param().ycoord();
    width_  = this->layer_param_.box_param().width();
    height_ = this->layer_param_.box_param().height();

    num_ = top.size();
    channels_ = bottom[0]->channels();   

    const BoxParameter& box_param = this->layer_param_.box_param();
    xcoord_.clear();
    std::copy(box_param.xcoord().begin(),
    box_param.xcoord().end(),
      std::back_inserter(xcoord_));

    ycoord_.clear();
    std::copy(box_param.ycoord().begin(),
    box_param.ycoord().end(),
      std::back_inserter(ycoord_));
    
}


template <typename Dtype>
void BoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   for (int i = 0; i < num_; ++i) {
      top[i]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                       height_, width_);
    }
}




/* copy only clipped region */
  template <typename Dtype>
  void BoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data(); 

  int height_bottom = bottom[0]->height();
  int width_bottom = bottom[0]->width();

  for (int n = 0; n < num_; n++) {

      Dtype* top_data = top[n]->mutable_cpu_data();
      int v = ycoord_[n];
      int w = xcoord_[n];
     for(int b = 0; b < bottom[0]->num(); b++){

      for(int c = 0; c< channels_; c++) {
        
        for(int h = v; h<v+height_; h++) {
          
          int index_bottom =  b*c*height_bottom*width_bottom+c*height_bottom*width_bottom + h*width_bottom + w;
          int index_top = b*c*height_*width_+c*height_*width_ + (h-v)*width_ + 0;
          
          for (int i = 0; i <width_; i++) {
            top_data[index_top + i] = bottom_data[index_bottom + i];
          }
        }
      }
    }
  }
  
}

/* copy only clipped region */

  template <typename Dtype>
  void BoxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      if (!propagate_down[0]) { return; }
      
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

      int height_bottom = bottom[0]->height();
      int width_bottom = bottom[0]->width();

      for (int n = 0; n < num_; n++) {

        const Dtype* top_diff = top[n]->cpu_diff();
        int v = ycoord_[n];
        int w = xcoord_[n];
        for (int b = 0; b < bottom[0]->num(); b++){
          
          for(int c = 0; c< channels_; c++) {
            
            for(int h = v; h<v+height_; h++) {
            
              int index_bottom = b*c*height_bottom*width_bottom+c*height_bottom*width_bottom + h*width_bottom + w;
              int index_top = b*c*height_*width_+c*height_*width_ + (h-v)*width_ + 0;
              for (int i = 0; i < width_; i++) {
                bottom_diff[index_bottom + i] = top_diff[index_top + i];
              }
            }
          }
        }
      }
  }

#ifdef CPU_ONLY
STUB_GPU(BoxLayer);
#endif
INSTANTIATE_CLASS(BoxLayer);
REGISTER_LAYER_CLASS(Box);
}
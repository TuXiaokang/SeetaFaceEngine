#include <iostream>
#include <new>

#include "face_identification.h"

#ifdef _WIN32
#define SEETA_API __declspec(dllexport)
#else
#define SEETA_API
#endif

struct Image{
    uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t channels;
};

struct LandMarks{
    int x[5], y[5];
};

struct Face{
    int left, top, right, bottom;
    double score;
    Face* next = nullptr;
};

typedef seeta::FaceIdentification Identifier;

extern "C"
{
    SEETA_API void* get_face_identifier(char* model_path){
        return new(std::nothrow) seeta::FaceIdentification(model_path);
    }

    SEETA_API Image* crop_face(void* ptr, Image* image, LandMarks* marks){
        Identifier* identifier = reinterpret_cast<Identifier*>(ptr);
        seeta::ImageData src_image_data(image->width, image->height);
        src_image_data.data         = image->data;
        src_image_data.num_channels = image->channels;

        int crop_width    = identifier->crop_width();
        int crop_height   = identifier->crop_height();
        int crop_channels = identifier->crop_channels();
        int count = crop_width * crop_height * crop_channels;

        seeta::ImageData dis_image_data(crop_width, crop_height);
        dis_image_data.data = new uint8_t[count];
        dis_image_data.num_channels = crop_channels;

        seeta::FacialLandmark llpoint[5];
        for(int i = 0; i < 5; i++){
            llpoint[i].x = marks->x[i] * 1.0;
            llpoint[i].y = marks->y[i] * 1.0; 
        }

        identifier->CropFace(src_image_data, llpoint, dis_image_data);
        
        Image* dis_image = new Image;
        dis_image->data     = dis_image_data.data;
        dis_image->width    = dis_image_data.width;
        dis_image->height   = dis_image_data.height;
        dis_image->channels = dis_image_data.num_channels; 
        return dis_image;
    }

    SEETA_API float* extract_feature(void* ptr, Image* image){
        Identifier* identifier = reinterpret_cast<Identifier*>(ptr);
        seeta::ImageData image_data(image->width, image->height);
        image_data.data = image->data;
        image_data.num_channels = image->channels;
        float* feature = new float[2048];
        identifier->ExtractFeature(image_data, feature);
        return feature;   
    }

    SEETA_API float* extract_feature_with_crop(void* ptr, Image* image, LandMarks* marks){
        Identifier* identifier = reinterpret_cast<Identifier*>(ptr);
        seeta::ImageData image_data(image->width, image->height);
        image_data.data = image->data;
        image_data.num_channels = image->channels;
        float* feature = new float[2048];
        seeta::FacialLandmark llpoint[5];
        for(int i = 0; i < 5; i++){
            llpoint[i].x = marks->x[i] * 1.0;
            llpoint[i].y = marks->y[i] * 1.0; 
        }
        identifier->ExtractFeatureWithCrop(image_data, llpoint, feature);
        return feature;
    }

    SEETA_API float calc_similarity(void* ptr, float *featA, float* featB){
        Identifier* identifier = reinterpret_cast<Identifier*>(ptr);
        return identifier->CalcSimilarity(featA, featB);
    }

    SEETA_API void free_image_data(Image* image){
        if(image != nullptr){
            delete [] image->data;
            delete image;
        }
    }

    SEETA_API void free_feature(float* feat){
        if(feat != nullptr){
            delete[] feat;
        }
    }

    SEETA_API void free_identifier(void* ptr){
        if(ptr != nullptr){
            Identifier* identifier = reinterpret_cast<Identifier*>(ptr);
            delete identifier;
        }
    }
}
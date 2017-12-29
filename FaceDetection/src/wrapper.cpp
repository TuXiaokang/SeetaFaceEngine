#include <iostream>
#include <new>

#include "face_detection.h"

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

struct Face{
    int left, top, right, bottom;
    double score;
    Face* next = nullptr;
};

typedef seeta::FaceDetection Detector;
extern "C"
{
    SEETA_API void* get_face_detector(char* model_path){
        return new(std::nothrow) seeta::FaceDetection(model_path);
    }

    SEETA_API void set_min_face_size(void* ptr, int set_min_face_size){
        Detector* detector = reinterpret_cast<Detector *>(ptr);
        detector->SetMinFaceSize(set_min_face_size);
    }

    SEETA_API void set_score_thresh(void* ptr, float score_thresh){
        Detector* detector = reinterpret_cast<Detector*>(ptr);
        detector->SetScoreThresh(score_thresh);
    }

    SEETA_API void set_image_pyramid_scale_factor(void* ptr, float scale_factor){
        Detector* detector = reinterpret_cast<Detector*>(ptr);
        detector->SetImagePyramidScaleFactor(scale_factor);
    }

    SEETA_API void set_window_step(void* ptr, int step_x, int step_y){
        Detector* detector = reinterpret_cast<Detector*>(ptr);
        detector->SetWindowStep(step_x, step_y);
    }

    SEETA_API Face* detect(void* ptr, Image* image){
        Detector* detector = reinterpret_cast<Detector*>(ptr);

        seeta::ImageData image_data(image->width, image->height);
        image_data.data = image->data;
        std::vector<seeta::FaceInfo> faces = detector->Detect(image_data);
	    int nb_faces = faces.size();

        Face* root = nullptr;
        for(int i = 0; i < nb_faces; ++ i){
            Face* now = new Face;
            now->next = root;
            now->left = faces[i].bbox.x;
            now->top = faces[i].bbox.y;
            now->right = faces[i].bbox.width + now->left;
            now->bottom = faces[i].bbox.height + now->top;
            now->score = faces[i].score;
            root = now;
        }
	    return root;
    }

    SEETA_API void free_face_list(Face* root){
        while(root!=nullptr){
            Face* ptr = root;
            root = root->next;
            delete ptr;
        }
    }

    SEETA_API void free_detector(void* ptr){
        if(ptr != nullptr){
            Detector* detector = reinterpret_cast<Detector*>(ptr);
            delete detector;
        }
    }
}

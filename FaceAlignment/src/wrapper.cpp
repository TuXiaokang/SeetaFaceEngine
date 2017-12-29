#include <iostream>
#include <new>

#include "face_alignment.h"

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

struct LandMarks{
    int x[5], y[5];
};

typedef seeta::FaceAlignment Aligner;
    
extern "C"
{
    SEETA_API void* get_face_aligner(char* model_path){
        return new(std::nothrow) Aligner(model_path);
    }

    SEETA_API LandMarks* align(void* ptr, Image* image, Face* face){
        Aligner* aligner = reinterpret_cast<Aligner*> (ptr);
        
        seeta::ImageData image_data(image->width, image->height);
        image_data.data = image->data;

        seeta::FacialLandmark points[5];
        seeta::FaceInfo face_info;
        
        face_info.bbox.x = face->left;
        face_info.bbox.y = face->top;
        face_info.bbox.width = face->right - face->left;
        face_info.bbox.height = face->bottom - face->top;
        face_info.score = face->score;

        aligner->PointDetectLandmarks(image_data, face_info, points);
        
        LandMarks* root = new LandMarks;
        for(int j = 0; j < 5; j++){
            root->x[j] = points[j].x;
            root->y[j] = points[j].y;
        }
        return root;
    }

    SEETA_API void free_landmarks(LandMarks* root){
        if(root != nullptr){
            delete root;
        }
    }

    SEETA_API void free_aligner(void* ptr){
        if(ptr != nullptr){
            Aligner* aligner = reinterpret_cast<Aligner*> (ptr);
            delete aligner;
        }
    }
}
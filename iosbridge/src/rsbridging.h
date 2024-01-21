#ifndef rsbridging_h
#define rsbridging_h

#include <stdint.h>
typedef struct IOSMTModel IOSMTModel;
typedef struct IOSWhisperModel IOSWhisperModel;
// Function declaration
IOSMTModel* iosmt_model_new(const char* path, _Bool gpu);
void iosmt_model_free(IOSMTModel* model);
const char* iosmt_model_inference(IOSMTModel* ptr, const char* input);
const char* iosmt_model_inference_new(IOSMTModel* ptr, const char* input,void (*predictionStringCallback)(const char*));
const char* rust_greeting(const char* to);
void inference_free(char *);

IOSWhisperModel* ios_whisper_model_new(const char* path, _Bool gpu);
void ios_whisper_model_inference(IOSWhisperModel* ptr, const char* languagetoken);
const char* ios_whisper_model_stop_record(IOSWhisperModel* ptr, const char* languagetoken);
void ios_whisper_model_record_play(IOSWhisperModel* ptr);
#endif /* rsbridging_h */

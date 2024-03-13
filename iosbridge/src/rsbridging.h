#ifndef rsbridging_h
#define rsbridging_h

#include <stdint.h>
typedef struct IOSMTModel IOSMTModel;
typedef struct IOSWhisperModel IOSWhisperModel;
// Function declaration
IOSMTModel* iosmt_model_new(const char* path, _Bool gpu);
void iosmt_model_free(IOSMTModel* model);
const char* iosmt_model_inference(IOSMTModel* ptr, const char* input);
const char* iosmt_model_inference_new(IOSMTModel* ptr, const char* input, void (*predictionStringCallback)(const char*));
const char* rust_greeting(const char* to);
void happy_test();
void string_free(char *);

IOSWhisperModel* ios_whisper_model_new(const char* path, _Bool gpu);
void ios_whisper_model_record(IOSWhisperModel* ptr);
void ios_whisper_model_stop_record(IOSWhisperModel* ptr);
const char* ios_whisper_model_detect_language(IOSWhisperModel* ptr);
const char* ios_whisper_model_inference(IOSWhisperModel* ptr, const char* languagetoken, void (*predictionStringCallback)(const char*));
void ios_whisper_model_record_play(IOSWhisperModel* ptr);

//mbart model
typedef struct IOSMBartModel IOSMBartModel;
IOSMBartModel* iosmbart_model_new(const char* path, _Bool gpu);
const char* iosmbart_model_inference_new(IOSMBartModel* ptr, const char* input, const char* src_lan, const char* target_lan, void (*predictionStringCallback)(const char*));
void iosmbart_model_free(IOSMBartModel* ptr);
#endif /* rsbridging_h */

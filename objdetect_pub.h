#ifndef ENTRYFUNC_H
#define ENTRYFUNC_H

#ifdef __cplusplus
extern "C"
    {
#endif

void objdetect_init
    (
    char *weight_file_path
    );

void objdetect_main
    (
    unsigned char *src,
    int srcw,
    int srch
    );

void objdetect_free
    (
    void
    );

#ifdef __cplusplus
    }
#endif

#endif // ENTRYFUNC_H

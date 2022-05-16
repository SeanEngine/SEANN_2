//
// Created by Dylan on 5/14/2022.
//

#ifndef SEANN_2_DATALOADER_CUH
#define SEANN_2_DATALOADER_CUH

#include "Dataset.cuh"
#include <cstdlib>
#include <fstream>

namespace seio {
    typedef unsigned char BYTE;

    void readBytes(BYTE *buffer, unsigned long size, const char* binPath);

    unsigned long getFileSize(const char* binPath);

    Dataset* fetchIDX(Dataset* dataset, const char* binPath, uint32 step, bool isLabel);

} // seio

#endif //SEANN_2_DATALOADER_CUH

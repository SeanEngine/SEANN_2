//
// Created by Dylan on 5/14/2022.
//

#include "DataLoader.cuh"
#define RGB_DECAY (1.0f/256.0f)
#define toLittleEndian(i) ((i>>24)&0xFF) | ((i>>8)&0xFF00) | ((i<<8)&0xFF0000) | ((i<<24)&0xFF000000)

namespace seio {

    void readBytes(BYTE *buffer, unsigned long size, const char* binPath){
        FILE *fp = fopen(binPath, "rb");
        assert(fp != nullptr);
        fread(buffer, sizeof(BYTE), size, fp);
        fclose(fp);
    }

    unsigned long getFileSize(const char* binPath){
        std::ifstream in(binPath, std::ifstream::ate | std::ifstream::binary);
        unsigned long size = in.tellg();
        in.close();
        logDebug(LOG_SEG_SEIO,"Preparing file \"" + string(binPath) + "\" : size = "
        + to_string(size), LOG_COLOR_LIGHT_YELLOW);
        return size;
    }

    Tensor* fetchOneHotLabel(const BYTE* buffer, uint32 offset, shape4 dims){
        Tensor* label = Tensor::declare(dims)->createHost();
        BYTE labelVal = buffer[offset];
        for(uint32 i = 0; i < label->dims.size; i++){
            label->elements[i] = (labelVal == i) ? 1.0f : 0.0f;
        }
        return label;
    }

    //The IDX data must be in NCHW arrangement
    Tensor* fetchBinImage(const BYTE* buffer, uint32 offset, shape4 dims){
        Tensor* image = Tensor::declare(dims)->createHost();
        for(uint32 i = 0; i < image->dims.size; i++){
            image->elements[i] = (float)buffer[offset + i] * RGB_DECAY;
        }
        return image;
    }

    void fetchIDXThread(int tid, int tc, Tensor*(*decode)(const BYTE*, uint32, shape4),
                            Dataset* set, BYTE* buf,  uint32 begOffset, uint32 step, bool isLabel){
        int start = tid * (int)(set->EPOCH_SIZE / tc);
        int end = tid == tc - 1 ? (int)set->EPOCH_SIZE : start + (int)(set->EPOCH_SIZE / tc);

        for(int i = start; i < end; i++){
            uint32 offset = begOffset + i * step;
            if(isLabel){
                set->dataset[i]->label = decode(buf, offset, set->labelShape);
            }else{
                set->dataset[i]->X = decode(buf, offset, set->dataShape);
            }
        }
    }

    Dataset* fetchIDX(Dataset* dataset, const char* binPath, uint32 step, bool isLabel){

        BYTE* buffer;
        unsigned long size = getFileSize(binPath);
        cudaMallocHost(&buffer, size);

        //load IDX format headers
        readBytes(buffer, size, binPath);
        uint32 magic = toLittleEndian(*(uint32*)buffer);
        uint32 numItems = toLittleEndian(*(uint32*)(buffer + 4));

        logDebug(LOG_SEG_SEIO,"Loading IDX file : magic = "
        + to_string(magic) + ", numItems = " + to_string(numItems));
        assert(numItems == (int)dataset->EPOCH_SIZE);

        //load dimension info:
        uint32 dimCount = magic & 0xFF;
        uint32 START_OFFSET = isLabel ? 8 : dimCount * 4 + 8;

        _alloc<CPU_THREADS>(fetchIDXThread, isLabel ? fetchOneHotLabel : fetchBinImage,
                            dataset, buffer, START_OFFSET, step, isLabel);
        logDebug(LOG_SEG_SEIO, "Fetched IDX file for : " + to_string(dataset->EPOCH_SIZE) + " items");

        cudaFreeHost(buffer);
        return dataset;
    }
} // seio
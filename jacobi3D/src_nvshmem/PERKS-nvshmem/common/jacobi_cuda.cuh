#ifndef PERKS_CUDA_HEADER
#define PERKS_CUDA_HEADER
//template<class REAL>

//this is where the aimed implementation located
template<class REAL>
int j3d_iterative(REAL*, int, int, int, REAL*, int, int, int, bool, bool, int, bool, bool getminHeight=false);

#define PERKS_DECLARE_INITIONIZATION_ITERATIVE(_type) \
    int j3d_iterative(_type*,int,int, int,_type*, int, int, int, bool, bool, int, bool,bool);

template<class REAL>int getMinWidthY(int , int, int, bool isDoubleTile=false);

#endif

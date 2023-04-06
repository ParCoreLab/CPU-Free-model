

#ifndef PERKS_REFERENCE_HEADER
#define PERKS_REFERENCE_HEADER
//template<class REAL>
//void jacobi(REAL*, int, int, REAL*);

// single step reference
template<class REAL>
void j3d_gold(REAL*, int, int, int, REAL*);
// iterative reference
template<class REAL>
void j3d_gold_iterative(REAL*, int, int, int, REAL*, int );


#define PERKS_DECLARE_INITIONIZATION_REFERENCE(_type) \
    void j3d_gold(_type*,int,int,int,_type*);

#define PERKS_DECLARE_INITIONIZATION_REFERENCE_ITERATIVE(_type) \
    void j3d_gold_iterative(_type*,int,int,int,_type*, int);

#endif
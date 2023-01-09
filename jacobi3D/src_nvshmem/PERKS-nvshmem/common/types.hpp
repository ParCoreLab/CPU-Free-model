
#ifndef PERKS_TYPES
#define PERKS_TYPES

#define PERKS_INITIALIZE_ALL_TYPE(_macro) \
    template _macro(float);\
    template _macro(double)

#define PERKS_INITIALIZE_ALL_TYPE_1ARG(_macro,halo) \
    template _macro(float,halo);\
    template _macro(double,halo)

#define PERKS_INITIALIZE_ALL_TYPE_2ARG(_macro,a,b) \
    template _macro(float,a,b);\
    template _macro(double,a,b)

#define PERKS_INITIALIZE_ALL_TYPE_3ARG(_macro,a,b,c) \
    template _macro(float,a,b,c);\
    template _macro(double,a,b,c)


#define PERKS_INITIALIZE_ALL_TYPE_4ARG(_macro,a,b,c,d) \
    template _macro(float,a,b,c,d);\
    template _macro(double,a,b,c,d)


#define PERKS_INITIALIZE_ALL_TYPE_5ARG(_macro,a,b,c,d,e) \
    template _macro(float,a,b,c,d,e);\
    template _macro(double,a,b,c,d,e)

#define PERKS_INITIALIZE_ALL_TYPE_6ARG(_macro,a,b,c,d,e,f) \
    template _macro(float,a,b,c,d,e,f);\
    template _macro(double,a,b,c,d,e,f)

#define PERKS_INITIALIZE_ALL_TYPE_7ARG(_macro,a,b,c,d,e,f,g) \
    template _macro(float,a,b,c,d,e,f,g);\
    template _macro(double,a,b,c,d,e,f,g)

#define PERKS_INITIALIZE_ALL_TYPE_8ARG(_macro,a,b,c,d,e,f,g,h) \
    template _macro(float,a,b,c,d,e,f,g,h);\
    template _macro(double,a,b,c,d,e,f,g,h)    
#endif

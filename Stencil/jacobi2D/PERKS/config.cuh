#ifndef TILE_X
    #define TILE_X (256)
#endif 
#ifndef RTILE_Y
    // #define RTILE_Y (8)
    #define RTILE_Y (8)
#endif

//minimal architecture is 600

// #if defined(js2d5pt)
#define HALO (1)
    // #define REG_FOLDER_Y (5)
    
// #elif defined(js2d9pt)
//     #define HALO (2)
//     #define REG_FOLDER_Y (10)
// #elif defined(js2d13pt)
//     #define HALO (3)
//     #define REG_FOLDER_Y (10)
// #elif defined(js2d17pt)
//     #define HALO (4)
//     #define REG_FOLDER_Y (10)
// #elif defined(js2d21pt)
//     #define HALO (5)
//     #define REG_FOLDER_Y (10)
// #elif defined(js2d25pt)
//     #define HALO (6)
//     #define REG_FOLDER_Y (10)
// #elif defined(jb2d9pt)
//     #define HALO (1)
//     #define BOX
//     #define REG_FOLDER_Y (0)
// #elif defined(jb2d25pt)
//     #define HALO (2)
//     #define BOX
//     #define REG_FOLDER_Y (0)
    
//#endif

#ifndef Halo 
    #define Halo HALO
#endif
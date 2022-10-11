#pragma once
#define star_shape (1)
#define box_shape (2)
#define type0_shape (3)
#define poisson_shape (4)
template<int halo, int shape, int bdim, int ipt, int registeramount, int arch,  bool useSM, class REAL>
struct regfolder
{
	static int const val = 0;
	static bool const spill = true;
};
template<>
struct regfolder<1,star_shape,128,8,128,700,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,128,700,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,128,700,false,float>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,128,700,true,float>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,128,700,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,128,700,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,128,700,false,float>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,128,700,true,float>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,128,700,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,16,128,700,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,128,700,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,16,128,700,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,700,true,double>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,700,false,float>
{
	static int const val = 15;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,700,true,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,256,700,false,float>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,256,700,true,float>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,700,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,700,false,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,700,true,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,256,700,false,float>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,256,700,true,float>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,700,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,700,true,double>
{
	static int const val = 2;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,700,false,float>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,700,true,float>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,16,256,700,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,700,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,700,true,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,700,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,700,true,float>
{
	static int const val = 9;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,16,256,700,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,700,true,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,700,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,700,true,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,128,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,128,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,700,true,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,700,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,700,true,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,128,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,128,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,700,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,700,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,700,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,700,true,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,256,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,256,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,700,false,double>
{
	static int const val = 2;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,700,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,700,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,700,true,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,256,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,256,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,700,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,700,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,700,true,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,128,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,128,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,700,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,700,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,700,true,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,128,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,128,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,700,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,700,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,700,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,700,true,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,256,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,256,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,700,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,700,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,700,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,700,true,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,256,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,256,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,700,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,700,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,700,true,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,128,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,128,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,700,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,700,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,700,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,700,true,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,128,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,128,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,700,false,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,700,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,700,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,700,true,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,256,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,256,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,700,false,double>
{
	static int const val = 2;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,700,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,700,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,700,true,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,256,700,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,256,700,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,128,800,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,128,800,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,128,800,true,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,128,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,128,800,true,float>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,128,800,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,128,800,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,128,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,128,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,128,800,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,128,800,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,16,128,800,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,128,800,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,128,800,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,16,128,800,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,800,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,800,true,double>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,800,false,float>
{
	static int const val = 15;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,8,256,800,true,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,256,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,128,16,256,800,true,float>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,800,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,800,true,double>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,800,false,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,8,256,800,true,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,256,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,star_shape,256,16,256,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,800,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,800,true,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,800,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,8,256,800,true,float>
{
	static int const val = 11;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,128,16,256,800,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,800,false,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,800,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,800,false,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,8,256,800,true,float>
{
	static int const val = 13;
	static bool const spill = false;
};
template<>
struct regfolder<2,star_shape,256,16,256,800,false,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,800,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,800,true,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,800,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,128,800,true,float>
{
	static int const val = 2;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,128,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,128,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,800,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,800,true,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,800,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,128,800,true,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,128,800,false,float>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,128,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,800,false,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,800,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,800,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,8,256,800,true,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,256,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,128,16,256,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,800,false,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,800,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,800,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,8,256,800,true,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,256,800,false,float>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<1,box_shape,256,16,256,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,800,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,800,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,800,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,128,800,true,float>
{
	static int const val = 8;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,128,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,128,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,800,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,800,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,800,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,128,800,true,float>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,128,800,false,float>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,128,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,800,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,800,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,800,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,8,256,800,true,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,256,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,128,16,256,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,800,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,800,true,double>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,800,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,8,256,800,true,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,256,800,false,float>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<1,type0_shape,256,16,256,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,800,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,800,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,800,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,128,800,true,float>
{
	static int const val = 9;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,128,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,128,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,800,false,double>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,800,true,double>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,800,false,float>
{
	static int const val = 1;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,128,800,true,float>
{
	static int const val = 0;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,128,800,false,float>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,128,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,800,false,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,800,true,double>
{
	static int const val = 4;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,800,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,8,256,800,true,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,256,800,false,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,128,16,256,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,800,false,double>
{
	static int const val = 3;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,800,true,double>
{
	static int const val = 5;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,800,false,float>
{
	static int const val = 12;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,8,256,800,true,float>
{
	static int const val = 14;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,256,800,false,float>
{
	static int const val = 7;
	static bool const spill = false;
};
template<>
struct regfolder<1,poisson_shape,256,16,256,800,true,float>
{
	static int const val = 6;
	static bool const spill = false;
};

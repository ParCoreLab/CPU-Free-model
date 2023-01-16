#pragma once
template<int halo, bool isstar, int registeramount, int arch,  bool useSM, class REAL, int tile=8>
struct regfolder
{
	static int const val = 0;
};
template<>
struct regfolder<1,true,128,700,false,double,8>
{
	static int const val = 3;
};
template<>
struct regfolder<1,true,128,700,true,double,8>
{
	static int const val = 3;
};
template<>
struct regfolder<1,true,128,700,false,float,8>
{
	static int const val = 6;
};
template<>
struct regfolder<1,true,128,700,false,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<1,true,128,700,true,float,8>
{
	static int const val = 6;
};
template<>
struct regfolder<1,true,128,700,true,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<2,true,128,700,false,double,8>
{
	static int const val = 3;
};
template<>
struct regfolder<2,true,128,700,true,double,8>
{
	static int const val = 2;
};
template<>
struct regfolder<2,true,128,700,false,float,8>
{
	static int const val = 6;
};
template<>
struct regfolder<2,true,128,700,false,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<2,true,128,700,true,float,8>
{
	static int const val = 5;
};
template<>
struct regfolder<2,true,128,700,true,float,16>
{
	static int const val = 1;
};
template<>
struct regfolder<3,true,128,700,false,double,8>
{
	static int const val = 3;
};
template<>
struct regfolder<3,true,128,700,true,double,8>
{
	static int const val = 1;
};
template<>
struct regfolder<3,true,128,700,false,float,8>
{
	static int const val = 6;
};
template<>
struct regfolder<3,true,128,700,false,float,16>
{
	static int const val = 1;
};
template<>
struct regfolder<3,true,128,700,true,float,8>
{
	static int const val = 2;
};
template<>
struct regfolder<3,true,128,700,true,float,16>
{
	static int const val = 0;
};
template<>
struct regfolder<4,true,128,700,false,double,8>
{
	static int const val = 2;
};
template<>
struct regfolder<4,true,128,700,true,double,8>
{
	static int const val = 0;
};
template<>
struct regfolder<4,true,128,700,false,float,8>
{
	static int const val = 3;
};
template<>
struct regfolder<4,true,128,700,false,float,16>
{
	static int const val = 1;
};
template<>
struct regfolder<4,true,128,700,true,float,8>
{
	static int const val = 0;
};
template<>
struct regfolder<5,true,128,700,false,double,8>
{
	static int const val = 2;
};
template<>
struct regfolder<5,true,128,700,false,float,8>
{
	static int const val = 3;
};
template<>
struct regfolder<5,true,128,700,false,float,16>
{
	static int const val = 0;
};
template<>
struct regfolder<6,true,128,700,false,double,8>
{
	static int const val = 1;
};
template<>
struct regfolder<6,true,128,700,false,float,8>
{
	static int const val = 1;
};
template<>
struct regfolder<6,true,128,700,false,float,16>
{
	static int const val = 0;
};
template<>
struct regfolder<1,true,256,700,false,double,8>
{
	static int const val = 11;
};
template<>
struct regfolder<1,true,256,700,true,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<1,true,256,700,false,float,8>
{
	static int const val = 24;
};
template<>
struct regfolder<1,true,256,700,false,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<1,true,256,700,true,float,8>
{
	static int const val = 15;
};
template<>
struct regfolder<1,true,256,700,true,float,16>
{
	static int const val = 8;
};
template<>
struct regfolder<2,true,256,700,false,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<2,true,256,700,true,double,8>
{
	static int const val = 9;
};
template<>
struct regfolder<2,true,256,700,false,float,8>
{
	static int const val = 15;
};
template<>
struct regfolder<2,true,256,700,false,float,16>
{
	static int const val = 8;
};
template<>
struct regfolder<2,true,256,700,true,float,8>
{
	static int const val = 21;
};
template<>
struct regfolder<2,true,256,700,true,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<3,true,256,700,false,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<3,true,256,700,true,double,8>
{
	static int const val = 9;
};
template<>
struct regfolder<3,true,256,700,false,float,8>
{
	static int const val = 15;
};
template<>
struct regfolder<3,true,256,700,false,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<3,true,256,700,true,float,8>
{
	static int const val = 22;
};
template<>
struct regfolder<3,true,256,700,true,float,16>
{
	static int const val = 8;
};
template<>
struct regfolder<4,true,256,700,false,double,8>
{
	static int const val = 9;
};
template<>
struct regfolder<4,true,256,700,true,double,8>
{
	static int const val = 8;
};
template<>
struct regfolder<4,true,256,700,false,float,8>
{
	static int const val = 23;
};
template<>
struct regfolder<4,true,256,700,false,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<4,true,256,700,true,float,8>
{
	static int const val = 21;
};
template<>
struct regfolder<4,true,256,700,true,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<5,true,256,700,false,double,8>
{
	static int const val = 8;
};
template<>
struct regfolder<5,true,256,700,true,double,8>
{
	static int const val = 7;
};
template<>
struct regfolder<5,true,256,700,false,float,8>
{
	static int const val = 15;
};
template<>
struct regfolder<5,true,256,700,false,float,16>
{
	static int const val = 8;
};
template<>
struct regfolder<5,true,256,700,true,float,8>
{
	static int const val = 19;
};
template<>
struct regfolder<5,true,256,700,true,float,16>
{
	static int const val = 8;
};
template<>
struct regfolder<6,true,256,700,false,double,8>
{
	static int const val = 4;
};
template<>
struct regfolder<6,true,256,700,true,double,8>
{
	static int const val = 6;
};
template<>
struct regfolder<6,true,256,700,false,float,8>
{
	static int const val = 15;
};
template<>
struct regfolder<6,true,256,700,false,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<6,true,256,700,true,float,8>
{
	static int const val = 14;
};
template<>
struct regfolder<6,true,256,700,true,float,16>
{
	static int const val = 8;
};
template<>
struct regfolder<1,false,128,700,false,double,8>
{
	static int const val = 3;
};
template<>
struct regfolder<1,false,128,700,true,double,8>
{
	static int const val = 2;
};
template<>
struct regfolder<1,false,128,700,false,float,8>
{
	static int const val = 6;
};
template<>
struct regfolder<1,false,128,700,false,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<1,false,128,700,true,float,8>
{
	static int const val = 5;
};
template<>
struct regfolder<1,false,128,700,true,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<2,false,128,700,false,double,8>
{
	static int const val = 1;
};
template<>
struct regfolder<2,false,128,700,true,double,8>
{
	static int const val = 0;
};
template<>
struct regfolder<2,false,128,700,false,float,8>
{
	static int const val = 6;
};
template<>
struct regfolder<2,false,128,700,false,float,16>
{
	static int const val = 1;
};
template<>
struct regfolder<2,false,128,700,true,float,8>
{
	static int const val = 2;
};
template<>
struct regfolder<2,false,128,700,true,float,16>
{
	static int const val = 0;
};
template<>
struct regfolder<1,false,256,700,false,double,8>
{
	static int const val = 11;
};
template<>
struct regfolder<1,false,256,700,true,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<1,false,256,700,false,float,8>
{
	static int const val = 24;
};
template<>
struct regfolder<1,false,256,700,false,float,16>
{
	static int const val = 10;
};
template<>
struct regfolder<1,false,256,700,true,float,8>
{
	static int const val = 21;
};
template<>
struct regfolder<1,false,256,700,true,float,16>
{
	static int const val = 9;
};
template<>
struct regfolder<2,false,256,700,false,double,8>
{
	static int const val = 9;
};
template<>
struct regfolder<2,false,256,700,true,double,8>
{
	static int const val = 7;
};
template<>
struct regfolder<2,false,256,700,false,float,8>
{
	static int const val = 21;
};
template<>
struct regfolder<2,false,256,700,false,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<2,false,256,700,true,float,8>
{
	static int const val = 20;
};
template<>
struct regfolder<2,false,256,700,true,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<1,true,128,800,false,double,8>
{
	static int const val = 5;
};
template<>
struct regfolder<1,true,128,800,true,double,8>
{
	static int const val = 5;
};
template<>
struct regfolder<1,true,128,800,false,float,8>
{
	static int const val = 12;
};
template<>
struct regfolder<1,true,128,800,false,float,16>
{
	static int const val = 4;
};
template<>
struct regfolder<1,true,128,800,true,float,8>
{
	static int const val = 11;
};
template<>
struct regfolder<1,true,128,800,true,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<2,true,128,800,false,double,8>
{
	static int const val = 4;
};
template<>
struct regfolder<2,true,128,800,true,double,8>
{
	static int const val = 4;
};
template<>
struct regfolder<2,true,128,800,false,float,8>
{
	static int const val = 11;
};
template<>
struct regfolder<2,true,128,800,false,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<2,true,128,800,true,float,8>
{
	static int const val = 7;
};
template<>
struct regfolder<2,true,128,800,true,float,16>
{
	static int const val = 1;
};
template<>
struct regfolder<3,true,128,800,false,double,8>
{
	static int const val = 4;
};
template<>
struct regfolder<3,true,128,800,true,double,8>
{
	static int const val = 2;
};
template<>
struct regfolder<3,true,128,800,false,float,8>
{
	static int const val = 10;
};
template<>
struct regfolder<3,true,128,800,false,float,16>
{
	static int const val = 4;
};
template<>
struct regfolder<3,true,128,800,true,float,8>
{
	static int const val = 9;
};
template<>
struct regfolder<3,true,128,800,true,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<4,true,128,800,false,double,8>
{
	static int const val = 3;
};
template<>
struct regfolder<4,true,128,800,true,double,8>
{
	static int const val = 0;
};
template<>
struct regfolder<4,true,128,800,false,float,8>
{
	static int const val = 7;
};
template<>
struct regfolder<4,true,128,800,false,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<4,true,128,800,true,float,8>
{
	static int const val = 9;
};
template<>
struct regfolder<4,true,128,800,true,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<5,true,128,800,false,double,8>
{
	static int const val = 2;
};
template<>
struct regfolder<5,true,128,800,false,float,8>
{
	static int const val = 3;
};
template<>
struct regfolder<5,true,128,800,false,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<5,true,128,800,true,float,8>
{
	static int const val = 8;
};
template<>
struct regfolder<5,true,128,800,true,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<6,true,128,800,false,double,8>
{
	static int const val = 1;
};
template<>
struct regfolder<6,true,128,800,false,float,8>
{
	static int const val = 1;
};
template<>
struct regfolder<6,true,128,800,false,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<6,true,128,800,true,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<1,true,256,800,false,double,8>
{
	static int const val = 13;
};
template<>
struct regfolder<1,true,256,800,true,double,8>
{
	static int const val = 13;
};
template<>
struct regfolder<1,true,256,800,false,float,8>
{
	static int const val = 27;
};
template<>
struct regfolder<1,true,256,800,false,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<1,true,256,800,true,float,8>
{
	static int const val = 27;
};
template<>
struct regfolder<1,true,256,800,true,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<2,true,256,800,false,double,8>
{
	static int const val = 12;
};
template<>
struct regfolder<2,true,256,800,true,double,8>
{
	static int const val = 12;
};
template<>
struct regfolder<2,true,256,800,false,float,8>
{
	static int const val = 26;
};
template<>
struct regfolder<2,true,256,800,false,float,16>
{
	static int const val = 10;
};
template<>
struct regfolder<2,true,256,800,true,float,8>
{
	static int const val = 25;
};
template<>
struct regfolder<2,true,256,800,true,float,16>
{
	static int const val = 11;
};
template<>
struct regfolder<3,true,256,800,false,double,8>
{
	static int const val = 12;
};
template<>
struct regfolder<3,true,256,800,true,double,8>
{
	static int const val = 11;
};
template<>
struct regfolder<3,true,256,800,false,float,8>
{
	static int const val = 25;
};
template<>
struct regfolder<3,true,256,800,false,float,16>
{
	static int const val = 11;
};
template<>
struct regfolder<3,true,256,800,true,float,8>
{
	static int const val = 25;
};
template<>
struct regfolder<3,true,256,800,true,float,16>
{
	static int const val = 8;
};
template<>
struct regfolder<4,true,256,800,false,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<4,true,256,800,true,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<4,true,256,800,false,float,8>
{
	static int const val = 23;
};
template<>
struct regfolder<4,true,256,800,false,float,16>
{
	static int const val = 11;
};
template<>
struct regfolder<4,true,256,800,true,float,8>
{
	static int const val = 24;
};
template<>
struct regfolder<4,true,256,800,true,float,16>
{
	static int const val = 10;
};
template<>
struct regfolder<5,true,256,800,false,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<5,true,256,800,true,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<5,true,256,800,false,float,8>
{
	static int const val = 19;
};
template<>
struct regfolder<5,true,256,800,false,float,16>
{
	static int const val = 11;
};
template<>
struct regfolder<5,true,256,800,true,float,8>
{
	static int const val = 22;
};
template<>
struct regfolder<5,true,256,800,true,float,16>
{
	static int const val = 10;
};
template<>
struct regfolder<6,true,256,800,false,double,8>
{
	static int const val = 7;
};
template<>
struct regfolder<6,true,256,800,true,double,8>
{
	static int const val = 8;
};
template<>
struct regfolder<6,true,256,800,false,float,8>
{
	static int const val = 15;
};
template<>
struct regfolder<6,true,256,800,false,float,16>
{
	static int const val = 11;
};
template<>
struct regfolder<6,true,256,800,true,float,8>
{
	static int const val = 15;
};
template<>
struct regfolder<6,true,256,800,true,float,16>
{
	static int const val = 10;
};
template<>
struct regfolder<1,false,128,800,false,double,8>
{
	static int const val = 4;
};
template<>
struct regfolder<1,false,128,800,true,double,8>
{
	static int const val = 4;
};
template<>
struct regfolder<1,false,128,800,false,float,8>
{
	static int const val = 11;
};
template<>
struct regfolder<1,false,128,800,false,float,16>
{
	static int const val = 4;
};
template<>
struct regfolder<1,false,128,800,true,float,8>
{
	static int const val = 10;
};
template<>
struct regfolder<1,false,128,800,true,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<2,false,128,800,false,double,8>
{
	static int const val = 3;
};
template<>
struct regfolder<2,false,128,800,true,double,8>
{
	static int const val = 2;
};
template<>
struct regfolder<2,false,128,800,false,float,8>
{
	static int const val = 9;
};
template<>
struct regfolder<2,false,128,800,false,float,16>
{
	static int const val = 3;
};
template<>
struct regfolder<2,false,128,800,true,float,8>
{
	static int const val = 8;
};
template<>
struct regfolder<2,false,128,800,true,float,16>
{
	static int const val = 2;
};
template<>
struct regfolder<1,false,256,800,false,double,8>
{
	static int const val = 12;
};
template<>
struct regfolder<1,false,256,800,true,double,8>
{
	static int const val = 12;
};
template<>
struct regfolder<1,false,256,800,false,float,8>
{
	static int const val = 27;
};
template<>
struct regfolder<1,false,256,800,false,float,16>
{
	static int const val = 12;
};
template<>
struct regfolder<1,false,256,800,true,float,8>
{
	static int const val = 24;
};
template<>
struct regfolder<1,false,256,800,true,float,16>
{
	static int const val = 11;
};
template<>
struct regfolder<2,false,256,800,false,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<2,false,256,800,true,double,8>
{
	static int const val = 10;
};
template<>
struct regfolder<2,false,256,800,false,float,8>
{
	static int const val = 23;
};
template<>
struct regfolder<2,false,256,800,false,float,16>
{
	static int const val = 7;
};
template<>
struct regfolder<2,false,256,800,true,float,8>
{
	static int const val = 23;
};
template<>
struct regfolder<2,false,256,800,true,float,16>
{
	static int const val = 8;
};

#include <stdio.h>
#include <string.h>

void IMG_quantize_c
(
    short           * data,
    unsigned short  num_blks,
    unsigned short  blk_size,
    const short     * recip_tbl,
    int             q_pt
);

/* ======================================================================== */
/*  Constant dataset.                                                       */
/* ======================================================================== */
#define BLK_SIZE (64)
#define NUM_BLKS (1)
#define Q_PT     (3)


/* ======================================================================== */
/*  Initialize arrays with random test data.                                */
/* ======================================================================== */
short  data_c[64] =
{
    -0x2EA8,  0x0A1B, -0x276E, -0x5A41, -0x23AB, -0x0B17,  0x3BD9,  0x3ED9,
     0x092E,  0x67EC, -0x1E61,  0x2473, -0x3C64, -0x31BC, -0x7B1E, -0x1F25,
     0x3701,  0x1FB6, -0x630F, -0x1161, -0x6E49,  0x2CFC, -0x6283,  0x7DA5,
     0x5D74,  0x1BE2,  0x47DF,  0x5AE4, -0x3365,  0x4095,  0x5F62,  0x0741,
     0x7FAC, -0x6399,  0x2E65,  0x554B,  0x2A43,  0x4F5F, -0x17CC, -0x316C,
    -0x69E6, -0x14CC, -0x6996, -0x3326,  0x54DC,  0x61A3,  0x709D, -0x28C6,
    -0x0C4C, -0x7D3D, -0x088A, -0x7774, -0x2121,  0x2F7F, -0x16D4, -0x6E5F,
    -0x1C09, -0x3B29,  0x4F66, -0x7B15, -0x6447,  0x47EA,  0x3DE5, -0x3445
};

int num_blks = 1;


const  short recip_tbl[] =
{
    0x1000,  0x1746,  0x1555,  0x1249,  0x1555,  0x199a,  0x1000,  0x1249,
    0x13b1,  0x1249,  0x0e39,  0x0f0f,  0x1000,  0x0d79,  0x0aab,  0x0666,
    0x09d9,  0x0aab,  0x0ba3,  0x0ba3,  0x0aab,  0x0539,  0x0750,  0x06eb,
    0x08d4,  0x0666,  0x046a,  0x0505,  0x0432,  0x0444,  0x047e,  0x0505,
    0x0492,  0x04a8,  0x0400,  0x038e,  0x02c8,  0x0348,  0x0400,  0x03c4,
    0x02f1,  0x03b6,  0x04a8,  0x0492,  0x0333,  0x0259,  0x0329,  0x02f1,
    0x02b2,  0x029d,  0x027c,  0x0276,  0x027c,  0x0421,  0x0353,  0x0244,
    0x021e,  0x0249,  0x028f,  0x0222,  0x02c8,  0x0289,  0x027c,  0x0296
};

short  data_c_expected[64] = { -20480, 26124, -9137, -18897, -7129, -32091, -19968, -23108, -26534, -30902, -563, -25608, 14336, 15941, -11553, 5928, -19173, 18770, -5874, -18296, -4248, 24068, -3006, -22897, 8322, 19714, -22581, 1871, 3145, 28531, -28856, -29399, -4093, 2035, 12928, -6637, -20149, -29193, 6656, -17238, 4175, 23135, -29774, -14453, -4229, -22795, 31749, 563, -9371, 5875, 22821, 16657, -18879, -31892, -31831, -16871, -27490, 6634, 25791, 12199, 9041, -13817, 14494, 6834 };

int main(int argc, char** argv)
{
        #pragma monitor start
        #pragma kernel
	IMG_quantize_c(data_c, num_blks, BLK_SIZE, recip_tbl, Q_PT);
        #pragma monitor stop

	if (argc > 42 && ! strcmp(argv[0], ""))	printf("%hd", data_c[64-1]);

	int i;
	for(i=0; i < 64; i++) {
			if(data_c[i] != data_c_expected[i]) {
					return 1;
			}
	}
	return 10;


}

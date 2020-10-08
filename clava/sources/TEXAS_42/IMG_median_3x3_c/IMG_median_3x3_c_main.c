#include <stdio.h>
#include <string.h>

void IMG_median_3x3_c
(
    const unsigned char * i_data,
    int n,
    unsigned char       * o_data
);

/* ======================================================================== */
/*  IMGLIB function-specific alignments. Refer to the                       */
/*  TMS320C64x IMG Library Programmer's Reference for details.              */
/* ======================================================================== */
#pragma DATA_ALIGN(in_data, 8);
#pragma DATA_ALIGN(out_data_c, 8);

/* ======================================================================== */
/*  Constant dataset.                                                       */
/* ======================================================================== */
#define COLS (256)
#define N    (COLS*3)


/* ======================================================================== */
/*  Initialize arrays with random test data.                                */
/* ======================================================================== */
unsigned char  in_data[N] =
{
     0xF1,  0x9A,  0xEC,  0x8F,  0x7A,  0xCD,  0x5C,  0x06,
     0x80,  0xCD,  0x2F,  0xE0,  0x49,  0xF0,  0x22,  0x2C,
     0x90,  0x4E,  0x1A,  0xE2,  0x12,  0xCC,  0x1D,  0x7E,
     0x47,  0x00,  0x18,  0x3D,  0x31,  0x94,  0x89,  0x0B,
     0x16,  0xD2,  0x05,  0x49,  0xB8,  0x24,  0x73,  0xBB,
     0xA3,  0xDD,  0x57,  0x61,  0x27,  0x7C,  0xA5,  0xD9,
     0x01,  0xC3,  0x4D,  0x55,  0xFD,  0x41,  0x1A,  0xD5,
     0xCE,  0x47,  0xF8,  0x21,  0x78,  0x2A,  0xD4,  0xE8,
     0x5C,  0x77,  0x2F,  0x87,  0x54,  0x67,  0x8D,  0x8F,
     0x4A,  0xA2,  0x48,  0x01,  0xDB,  0x6B,  0xB5,  0x54,
     0x7F,  0x9F,  0x60,  0x52,  0x48,  0x71,  0x7F,  0xCF,
     0xD1,  0xAC,  0x31,  0x33,  0xDE,  0x4C,  0x12,  0xE0,
     0x2E,  0x11,  0x13,  0xAC,  0x08,  0xD1,  0xFA,  0xB1,
     0xB0,  0x82,  0x78,  0x2B,  0x1B,  0x16,  0xD4,  0x7C,
     0x6C,  0xC9,  0x64,  0xE0,  0x77,  0x7A,  0xAE,  0x07,
     0x8A,  0xED,  0xAC,  0xA4,  0x06,  0x0D,  0x1E,  0x72,
     0xBC,  0x66,  0x88,  0x86,  0x46,  0x74,  0x26,  0x36,
     0xCA,  0x6A,  0xCC,  0x1B,  0x7A,  0xEA,  0xE4,  0x43,
     0xAE,  0x9C,  0x84,  0xD7,  0xBF,  0x42,  0xF4,  0xA8,
     0x55,  0x83,  0x01,  0x11,  0x4E,  0x39,  0xAF,  0x94,
     0xE9,  0x75,  0x4F,  0xBE,  0x56,  0x5B,  0x02,  0x5D,
     0x45,  0x6B,  0xFC,  0x9E,  0x9B,  0x1B,  0x66,  0xD1,
     0x0D,  0x4F,  0x97,  0x60,  0x12,  0x64,  0x9A,  0x58,
     0xA5,  0xF9,  0xBE,  0x24,  0x81,  0x25,  0x3D,  0x58,
     0x2E,  0xEE,  0x9D,  0xFF,  0x74,  0xE4,  0xDA,  0xD0,
     0x97,  0x6F,  0xEA,  0x31,  0x85,  0x3A,  0x34,  0x53,
     0xB4,  0x1A,  0xC9,  0x6B,  0x6A,  0xA3,  0x99,  0x82,
     0x47,  0xA3,  0xE8,  0xD5,  0xE9,  0xB3,  0xB5,  0x8E,
     0xE3,  0x9D,  0x00,  0x67,  0x69,  0xF0,  0x06,  0x83,
     0x98,  0x8B,  0x6B,  0x04,  0x8C,  0x35,  0x49,  0xFF,
     0x64,  0x91,  0xD6,  0xB6,  0xD2,  0x70,  0x6F,  0x65,
     0x76,  0x2B,  0x34,  0xCA,  0x34,  0xF1,  0x6F,  0xC6,
     0x79,  0x58,  0x17,  0xB7,  0x22,  0x53,  0x09,  0x9E,
     0x2B,  0x6E,  0x0D,  0x5D,  0x5A,  0x44,  0x05,  0xCB,
     0xD2,  0xB6,  0x5C,  0xC5,  0xAC,  0xEC,  0xA0,  0x78,
     0x36,  0x77,  0x70,  0x60,  0xC4,  0xE9,  0xE4,  0x57,
     0x1B,  0xD8,  0x98,  0xBE,  0xC5,  0x16,  0x8D,  0xFC,
     0x70,  0x5D,  0x45,  0xDC,  0x9D,  0x39,  0x7D,  0x4B,
     0xB6,  0x48,  0xC6,  0x65,  0xB3,  0xFA,  0x55,  0xA1,
     0x82,  0x69,  0xA5,  0xC8,  0xCF,  0x1C,  0xB6,  0xC3,
     0x47,  0x19,  0x3B,  0x8C,  0xC0,  0xF7,  0xCB,  0xB5,
     0x06,  0x52,  0xA1,  0x1E,  0xC3,  0xBE,  0x4B,  0x4F,
     0xFE,  0xCA,  0x40,  0x2D,  0x3B,  0xDE,  0xA7,  0xBA,
     0xC1,  0xF8,  0x69,  0xD3,  0xE3,  0x1D,  0x6C,  0x1B,
     0xC5,  0xAA,  0xE8,  0xE6,  0x59,  0xBC,  0xD5,  0x40,
     0xDC,  0xB1,  0x61,  0x44,  0x62,  0xFB,  0x65,  0xA2,
     0xB8,  0xBF,  0x50,  0x7F,  0x2C,  0xDC,  0x08,  0xE8,
     0x0D,  0x20,  0xEC,  0xD5,  0x48,  0xE0,  0x82,  0xCC,
     0x9B,  0x78,  0xC7,  0x72,  0x35,  0xC8,  0x41,  0xA0,
     0x0A,  0x18,  0x10,  0xDC,  0xB1,  0x21,  0x7E,  0x8A,
     0x26,  0xA7,  0x77,  0x8D,  0x3B,  0x66,  0x59,  0x43,
     0xC4,  0x2C,  0x34,  0x29,  0xBB,  0x28,  0x35,  0x61,
     0x62,  0x95,  0x9E,  0x33,  0x66,  0x80,  0xC4,  0x27,
     0x3A,  0xC2,  0x8E,  0xA4,  0xA5,  0x65,  0x60,  0x69,
     0x77,  0x2B,  0x6B,  0xD9,  0xE2,  0x51,  0x1B,  0x50,
     0xE8,  0xDF,  0x16,  0xE3,  0x06,  0xA8,  0x95,  0xB3,
     0x8D,  0x74,  0x08,  0xB5,  0x6C,  0xEE,  0x21,  0x46,
     0x1F,  0x09,  0x0B,  0xB7,  0x2A,  0xDA,  0xD7,  0x1E,
     0xBC,  0x0A,  0x6D,  0xC0,  0x96,  0x93,  0x4E,  0xCC,
     0xCB,  0xE7,  0x59,  0x0D,  0xF9,  0x41,  0x64,  0x0C,
     0x1D,  0x21,  0xFF,  0x8A,  0xB5,  0x3C,  0x13,  0xF2,
     0x74,  0x7F,  0x7B,  0x95,  0xF8,  0xC4,  0x9C,  0x89,
     0x7C,  0x3F,  0x24,  0x7B,  0xAF,  0x0C,  0x70,  0x45,
     0x85,  0x3B,  0x99,  0x1C,  0x3F,  0xF5,  0x2A,  0xBC,
     0x40,  0xEE,  0x39,  0x7B,  0x21,  0x6A,  0x85,  0x13,
     0x11,  0x1F,  0x04,  0x8C,  0x83,  0xBE,  0xC2,  0xCF,
     0x7F,  0xD6,  0xA2,  0x28,  0x9E,  0xD4,  0x26,  0x06,
     0xBF,  0xC6,  0x57,  0xF3,  0x88,  0xFB,  0x22,  0x9A,
     0x60,  0x1F,  0xE1,  0x03,  0xF9,  0xD4,  0xC0,  0x8E,
     0x81,  0x1B,  0xD4,  0x0A,  0x8E,  0x61,  0x30,  0xBE,
     0x3C,  0x09,  0xE9,  0x51,  0xD4,  0x96,  0xFE,  0xD8,
     0x42,  0xED,  0xC3,  0x18,  0x9F,  0x80,  0x5A,  0xFB,
     0x1A,  0x0C,  0x1A,  0x86,  0xEC,  0x02,  0xF7,  0x57,
     0xB2,  0xBD,  0xF8,  0x40,  0xD2,  0xBD,  0x96,  0xF2,
     0x7F,  0x08,  0x96,  0x79,  0xE7,  0x71,  0xE9,  0x0C,
     0xD9,  0x73,  0xB8,  0x77,  0xCF,  0x8D,  0xA4,  0x9B,
     0xAB,  0x38,  0xE4,  0xC9,  0x82,  0x3D,  0xB9,  0x19,
     0x25,  0xE2,  0xA4,  0xBA,  0x4B,  0x50,  0x71,  0x21,
     0xB7,  0x04,  0x15,  0x1F,  0x7B,  0xE5,  0xC7,  0x5A,
     0xFF,  0x2F,  0x6E,  0x58,  0x2E,  0x9D,  0xC4,  0xEF,
     0x3D,  0xE6,  0xCB,  0x32,  0x90,  0x6F,  0xC6,  0xEF,
     0x38,  0x96,  0x6D,  0x48,  0x9F,  0xD6,  0x8B,  0x1A,
     0x6A,  0xEF,  0xF0,  0xCF,  0x61,  0x46,  0xA6,  0x16,
     0xD4,  0x22,  0x62,  0xED,  0x3C,  0x53,  0x6C,  0x73,
     0xAE,  0xAB,  0xF0,  0x3C,  0x27,  0xCB,  0xCC,  0xAA,
     0xF2,  0x40,  0xFC,  0xCD,  0x1C,  0x88,  0x87,  0xF8,
     0x67,  0x73,  0xB6,  0xAE,  0x74,  0x3C,  0x1C,  0x2C,
     0xEA,  0x5E,  0xF1,  0x00,  0x6E,  0xE0,  0x72,  0x60,
     0x3E,  0x6A,  0xE6,  0xBB,  0x82,  0xC3,  0x8B,  0x2C,
     0xE0,  0xE6,  0xB5,  0x74,  0xDE,  0x48,  0xEE,  0x03,
     0x12,  0xA6,  0xEC,  0x00,  0xB8,  0xC4,  0x4F,  0x74,
     0x56,  0x4F,  0x2A,  0x45,  0x04,  0x91,  0xD5,  0x58,
     0xA4,  0x5B,  0xE9,  0x52,  0x70,  0xBC,  0xD6,  0x52,
     0x85,  0x16,  0xD2,  0xBE,  0x88,  0x2F,  0x43,  0xC3,
     0x68,  0xF6,  0x9F,  0x3F,  0x2E,  0x44,  0x19,  0xD7,
     0x91,  0xD6,  0x54,  0xCD,  0xC8,  0xFF,  0x00,  0x7F
};

unsigned char  out_data_c[COLS];

unsigned char  out_data_c_expected[COLS] = { 127, 127, 121, 143, 122, 122, 92, 92, 43, 43, 43, 93, 90, 131, 90, 190, 144, 182, 144, 162, 158, 197, 160, 126, 71, 119, 87, 96, 96, 148, 148, 148, 87, 87, 96, 152, 184, 184, 184, 142, 142, 142, 112, 93, 97, 97, 124, 124, 125, 75, 77, 85, 179, 150, 179, 161, 161, 161, 165, 165, 165, 120, 128, 182, 182, 92, 47, 59, 134, 135, 192, 143, 143, 143, 161, 82, 161, 189, 189, 150, 127, 127, 127, 96, 82, 113, 127, 167, 193, 193, 184, 119, 184, 141, 141, 108, 155, 155, 170, 172, 172, 172, 185, 185, 177, 176, 164, 130, 97, 75, 98, 113, 124, 162, 108, 100, 100, 123, 123, 174, 138, 90, 138, 164, 110, 88, 72, 157, 155, 155, 155, 134, 134, 114, 111, 116, 65, 106, 106, 106, 122, 159, 159, 138, 126, 138, 156, 167, 141, 102, 97, 89, 166, 85, 85, 44, 60, 60, 78, 97, 115, 148, 158, 149, 86, 91, 102, 128, 93, 93, 142, 164, 164, 155, 102, 105, 105, 105, 107, 115, 151, 100, 81, 80, 88, 165, 223, 190, 110, 110, 114, 114, 96, 106, 116, 181, 157, 187, 139, 195, 139, 111, 151, 116, 133, 116, 133, 72, 83, 30, 166, 109, 150, 150, 150, 147, 116, 130, 89, 89, 89, 145, 179, 142, 142, 91, 157, 103, 112, 112, 112, 131, 131, 131, 127, 127, 140, 140, 136, 137, 124, 137, 124, 145, 159, 112, 111, 101, 112, 118, 118, 84, 84, 202, 111, 188 };

int main(int argc, char** argv)
{
        #pragma monitor start
        #pragma kernel
	IMG_median_3x3_c(in_data, COLS, out_data_c);
        #pragma monitor stop

	if (argc > 42 && ! strcmp(argv[0], ""))	printf("%u", (unsigned int) out_data_c[COLS-1]);

	int i;
	for(i=0; i < COLS; i++) {
			if(out_data_c[i] != out_data_c_expected[i]) {
					return 1;
			}
	}
	return 10;

}
#include <stdio.h>
#include <string.h>

void DSP_neg32_c
(
    const int    *x,
    int * r,
    int           nx
);

#pragma DATA_ALIGN(x_c, 8);
#pragma DATA_ALIGN(r_c, 8);

/* ======================================================================== */
/*  Constant dataset.                                                       */
/* ======================================================================== */
#define N    (256)


/* ======================================================================== */
/*  Initialize arrays with random test data.                                */
/* ======================================================================== */
int  x_c[N] =
{
     0x3104521E,  0x4B8AD6BF,  0x5B22DF04, -0x1CB3E263,
    -0x29CBF198, -0x59BB84A2,  0x69A21107, -0x77C3D2AC,
    -0x5FAE0476, -0x4964FB64,  0x4170AE36, -0x38CF2BE9,
    -0x66E180CD, -0x3407FF54,  0x3CC01B95,  0x598620BE,
     0x17DD1F86, -0x515F4FE5,  0x40B1B483,  0x29F23F7F,
     0x3DBC5B8A, -0x717F354D, -0x5273A0F9,  0x6742479A,
     0x7FBDE01E,  0x7B37E91F,  0x049AECAE, -0x75BF7D21,
     0x27B39148,  0x42B85805, -0x7B5D8D83, -0x6D3C9BC3,
     0x24308447, -0x3E56600B,  0x19EF2642, -0x05070664,
    -0x3E926933, -0x701C029E, -0x67E15F67, -0x14C7372C,
     0x47857F14, -0x597BEC57, -0x0E76BE66, -0x5A1A685C,
     0x4F3B8E6B,  0x7F7C9015,  0x1B5CBAD2,  0x0C192BE7,
    -0x05D52DAE,  0x3139DC61,  0x0199A7F1, -0x695953AA,
    -0x4BDADB1F,  0x7E6CFC1F, -0x7BC52569, -0x6471BAEC,
    -0x17BC8701, -0x0B6CC065, -0x4F6243D0, -0x4E06C164,
     0x50085EB7,  0x592BC591, -0x3E2D0D35, -0x3FCDEBF8,
     0x286037F7,  0x36793CE0,  0x31DEF547, -0x643A2D75,
    -0x0F8C5536,  0x21FF1919,  0x5420F64C,  0x5F13F8DB,
     0x7B9B47D3, -0x0965B6BF,  0x2A8B93B4, -0x14884DB2,
    -0x02A521B0, -0x7866BFFF, -0x44F485E7, -0x5402CA52,
    -0x0FBC7FD3,  0x3796C6E7, -0x22E31853,  0x0E8B55BE,
     0x0C5A326A, -0x1C85FE6A, -0x1CEF634B, -0x0C5591BE,
     0x35CB7566, -0x15E0EE4B, -0x13B17CD9,  0x335380EE,
     0x1EB6658A, -0x6CBDD4DE, -0x3FDCBB00, -0x5F552B87,
     0x65719291, -0x69F791F2,  0x3EF1B38B, -0x109BD2E7,
    -0x1A3EF1E1, -0x1E428B78,  0x297AC6AC,  0x017FADFE,
    -0x3BA85289, -0x3D61B174,  0x63960365, -0x6296071A,
    -0x507D2678, -0x27482740, -0x177FEA25, -0x295530B8,
     0x3970C54F, -0x5336026F, -0x09733814,  0x015E26AB,
    -0x4E27284F, -0x37FA1190, -0x7940F4D9, -0x65E0B80C,
     0x0E9F6C9D,  0x6F63648C, -0x384DF289,  0x79D5D866,
    -0x55AFBC59,  0x40D3E712, -0x35A3B37A,  0x12DC54EB,
     0x08EC657B,  0x4AD778C8,  0x49438246, -0x31FCF19B,
     0x17F92D2D,  0x7053A139, -0x026658E0,  0x18462D39,
    -0x5A78CC45, -0x14594246, -0x3BC683D6, -0x31DB6BBC,
    -0x6F79CE88, -0x43EC4E3E,  0x0FF4171A, -0x7FD4308B,
    -0x246C156C,  0x2CAF97BF, -0x3420865B, -0x0F1F4C50,
    -0x013CED8D,  0x3FD64BD5,  0x57ADCBB2, -0x74C0BD62,
     0x7D6B7AD0, -0x6EE9B57F,  0x38D9241E, -0x65BC3B70,
     0x08375326, -0x586736DC,  0x3EDF03FA, -0x55B6CD24,
     0x350FB731,  0x36940925, -0x79CCF276,  0x29147AAB,
     0x16800984, -0x5B88244B,  0x7CE4CD84,  0x5229F025,
    -0x63F535B6, -0x4894E830, -0x77D4CF20,  0x6522978A,
     0x1A5D3079, -0x7B5C75B2,  0x7B69411F,  0x1159FAED,
    -0x052A83F2,  0x76D4A619, -0x3C2585D5, -0x15618CA4,
     0x4C0D4FD9,  0x4F10F27F,  0x78811B9C,  0x7E8C4D0B,
    -0x1CC853CC, -0x5153D6B2,  0x70687ACD, -0x08A6AA0B,
    -0x052C2712, -0x22561CF3, -0x7F10DFE6,  0x5E9B9EDF,
    -0x1D488206,  0x3055CCB1, -0x22C34E07,  0x3440206E,
    -0x5B281DA3, -0x1618586B, -0x12DFCCD8,  0x0BF78338,
     0x2B5A73D3, -0x7CC7FDD8,  0x5253CCC4,  0x375E50F5,
     0x68C3E981,  0x1D3AF3F4,  0x12958841,  0x377D3FF2,
     0x6073566A, -0x7ADA7FD6, -0x21CC676A,  0x0A317548,
     0x3AA01FC5,  0x6A712145, -0x03001A2D,  0x2DA59B4A,
     0x3D29F3A7,  0x2FCB8905,  0x58C352EE, -0x7EF12B1C,
     0x0152591A, -0x4299DC0D, -0x57237D05,  0x1BA4CFF1,
     0x27B6119F,  0x67D9141F, -0x31067872, -0x29FB5573,
     0x502C2B67,  0x6A82F0BC, -0x1EA95041,  0x2663D584,
     0x7B1FB3CE,  0x1A023D19, -0x036054DD,  0x0C93E9D1,
     0x26C6D0D8,  0x1258C03D,  0x18E91229, -0x1738F778,
    -0x51C16857, -0x0E1B90B7, -0x15DCE26F,  0x7660F35D,
    -0x45077F36, -0x149CC9EA, -0x436B07C6,  0x68262260,
     0x348D3541,  0x1322DA81, -0x38CBA559,  0x18F2FD9F,
     0x42899293, -0x6CAFB395,  0x7688EA08, -0x2A89274D
};

int  r_c[N];

int  r_c_expected[N] = { -822366750, -1267390143, -1529011972, 481550947, 701231512, 1505461410, -1772228871, 2009322156, 1605239926, 1231354724, -1097903670, 953101289, 1726054605, 872939348, -1019222933, -1501962430, -400367494, 1365200869, -1085387907, -703741823, -1035754378, 1904162125, 1383309561, -1732396954, -2143150110, -2067261727, -77261998, 1975483681, -666079560, -1119377413, 2069728643, 1832688579, -607159367, 1045848075, -435103298, 84346468, 1049782579, 1880883870, 1742823271, 348600108, -1199931156, 1501293655, 242663014, 1511680092, -1329303147, -2138869781, -459061970, -202976231, 97856942, -825875553, -26847217, 1767461802, 1272634143, -2121071647, 2076517737, 1685175020, 398231297, 191676517, 1331839952, 1309065572, -1342725815, -1496040849, 1043139893, 1070459896, -677394423, -913915104, -836695367, 1681534325, 260855094, -570366233, -1411446348, -1595144411, -2073774035, 157660863, -713790388, 344477106, 44376496, 2019999743, 1156875751, 1409469010, 264011731, -932628199, 585308243, -244012478, -207237738, 478543466, 485450571, 206934462, -902526310, 367062603, 330398937, -861110510, -515270026, 1824380126, 1071430400, 1599417223, -1701941905, 1777832434, -1056027531, 278647527, 440332769, 507677560, -695912108, -25144830, 1000886921, 1029812596, -1670775653, 1653999386, 1350379128, 659040064, 394258981, 693448888, -963691855, 1396048495, 158545940, -22947499, 1311189071, 939135376, 2034300121, 1709225996, -245329053, -1868784780, 944632457, -2044057702, 1437580377, -1087629074, 899920762, -316429547, -149710203, -1255635144, -1229161030, 838660507, -402205997, -1884528953, 40261856, -407252281, 1517866053, 341393990, 1002865622, 836463548, 1870253704, 1139559998, -267654938, 2144612491, 611063148, -749705151, 874546779, 253709392, 20770189, -1071008725, -1471007666, 1958788450, -2104195792, 1860810111, -953754654, 1706834800, -137843494, 1483159260, -1054802938, 1438043428, -890222385, -915671333, 2043474550, -689207979, -377489796, 1535648843, -2095369604, -1378480165, 1677014454, 1217718320, 2010435360, -1696765834, -442314873, 2069657010, -2070495519, -291109613, 86672370, -1993647641, 1009092053, 358714532, -1275940825, -1326510719, -2021727132, -2123123979, 482890700, 1364448946, -1885895373, 145140235, 86779666, 576068851, 2131812326, -1587257055, 491291142, -810929329, 583224839, -876617838, 1529355683, 370694251, 316656856, -200770360, -727348179, 2093481432, -1381223620, -928927989, -1757669761, -490402804, -311789633, -930955250, -1618171498, 2061139926, 567043946, -171013448, -983572421, -1785798981, 50338349, -765827914, -1026159527, -801868037, -1489195758, 2129734428, -22173978, 1117379597, 1461943557, -463785969, -666243487, -1742279711, 822507634, 704337267, -1345071975, -1786966204, 514412609, -644076932, -2065675214, -436354329, 56644829, -211020241, -650563800, -307806269, -417927721, 389609336, 1371629655, 236687543, 366797423, -1986065245, 1158119222, 345819626, 1131087814, -1747329632, -881669441, -321051265, 952870233, -418577823, -1116312211, 1823454101, -1988684296, 713631565 };


int main(int argc, char** argv)
{
        #pragma monitor start
        #pragma kernel
	DSP_neg32_c(x_c, r_c, N);
        #pragma monitor stop

	if (argc > 42 && ! strcmp(argv[0], ""))	printf("%d", r_c[N-1]);

	int i;
	for(i=0; i < N; i++) {
			if(r_c[i] != r_c_expected[i]) {
					return 1;
			}
	}
	return 10;

}

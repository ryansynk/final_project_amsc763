#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "test.h"
#include "matrix.h"
#include "qr.h"

bool assert_close(double actual, double expected, double rtol, double atol) {
    double abs_err = fabs(actual - expected);
    double rel_err = abs_err / expected;
    
    if (abs_err > atol) {
        return false;
    } else if (rel_err > rtol) {
        return false;
    } else {
        return true;
    }
}

bool assert_allclose_ptr(double *actual, int actual_rows, int actual_cols, 
                         double *expected, int expected_rows, int expected_cols, 
                         double rtol, double atol) {
    if (actual == NULL || expected == NULL) {
        return false;
    } 
    for (int i = 0; i < actual_rows; i++) {
        for (int j = 0; j < actual_cols; j++) {
            if (assert_close(actual[i + j * actual_rows], expected[i + j * expected_rows], rtol, atol) == false) {
                return false;
            }
        }
    }
    return true;
}

bool assert_allclose(Matrix *actual, Matrix *expected, double rtol, double atol) {
    if (actual == NULL || expected == NULL) {
        return false;
    } else if (actual->rows != expected->rows || actual->cols != expected->cols) {
        return false;
    } else {
        for (int i = 0; i < actual->rows; i++) {
            for (int j = 0; j < actual->cols; j++) {
                if (assert_close(actual->data[i][j], expected->data[i][j], rtol, atol) == false) {
                    return false;
                }
            }
        }
        return true;
    }
}

bool test_gemm() {

    Matrix *A = NULL;
    Matrix *B = NULL;
    Matrix *C = NULL;
    Matrix *D = NULL;
    bool result = false;

    init_matrix(&A, 7, 13);
    init_matrix(&B, 13, 11);
    init_matrix(&C, 7, 11);
    init_matrix(&D, 7, 11);

    double atol = 0.0000001;
    double rtol = 0.0000001;

    A->data[0][0] = 0.09542479; A->data[0][1] = 0.14712593; A->data[0][2] = 0.88998051; A->data[0][3] = 0.63738239; A->data[0][4] = 0.60919847; A->data[0][5] = 0.63385423; A->data[0][6] = 0.07044193; A->data[0][7] = 0.80001731; A->data[0][8] = 0.57447377; A->data[0][9] = 0.41220604; A->data[0][10] = 0.44199716; A->data[0][11] = 0.34032526; A->data[0][12] = 0.30402116;
    A->data[1][0] = 0.05253944; A->data[1][1] = 0.3099338 ; A->data[1][2] = 0.41934822; A->data[1][3] = 0.21770562; A->data[1][4] = 0.71618029; A->data[1][5] = 0.04691743; A->data[1][6] = 0.01123448; A->data[1][7] = 0.58888321; A->data[1][8] = 0.90963758; A->data[1][9] = 0.36686079; A->data[1][10] = 0.71321206; A->data[1][11] = 0.62204195; A->data[1][12] = 0.26497803;
    A->data[2][0] = 0.18519507; A->data[2][1] = 0.86807473; A->data[2][2] = 0.28382064; A->data[2][3] = 0.02517163; A->data[2][4] = 0.8736909 ; A->data[2][5] = 0.82186416; A->data[2][6] = 0.02038312; A->data[2][7] = 0.76401447; A->data[2][8] = 0.03913221; A->data[2][9] = 0.76419994; A->data[2][10] = 0.10807324; A->data[2][11] = 0.7201944 ; A->data[2][12] = 0.45104667;
    A->data[3][0] = 0.99036225; A->data[3][1] = 0.39912427; A->data[3][2] = 0.04770133; A->data[3][3] = 0.52822626; A->data[3][4] = 0.56031853; A->data[3][5] = 0.99224872; A->data[3][6] = 0.83712788; A->data[3][7] = 0.58753871; A->data[3][8] = 0.55986886; A->data[3][9] = 0.1237992 ; A->data[3][10] = 0.15090809; A->data[3][11] = 0.65082056; A->data[3][12] = 0.8029521 ;
    A->data[4][0] = 0.95523478; A->data[4][1] = 0.02029638; A->data[4][2] = 0.20414547; A->data[4][3] = 0.58910318; A->data[4][4] = 0.38610546; A->data[4][5] = 0.91633644; A->data[4][6] = 0.43657481; A->data[4][7] = 0.59146723; A->data[4][8] = 0.95725668; A->data[4][9] = 0.83482456; A->data[4][10] = 0.79017214; A->data[4][11] = 0.45145973; A->data[4][12] = 0.64938558;
    A->data[5][0] = 0.93957926; A->data[5][1] = 0.06003473; A->data[5][2] = 0.79513834; A->data[5][3] = 0.13106148; A->data[5][4] = 0.93389421; A->data[5][5] = 0.9893287 ; A->data[5][6] = 0.32802662; A->data[5][7] = 0.58543828; A->data[5][8] = 0.77108273; A->data[5][9] = 0.48034382; A->data[5][10] = 0.05005683; A->data[5][11] = 0.71144019; A->data[5][12] = 0.18448666;
    A->data[6][0] = 0.47556832; A->data[6][1] = 0.53139942; A->data[6][2] = 0.95901975; A->data[6][3] = 0.00252595; A->data[6][4] = 0.95524412; A->data[6][5] = 0.27680841; A->data[6][6] = 0.60158878; A->data[6][7] = 0.38262372; A->data[6][8] = 0.7563717 ; A->data[6][9] = 0.16674703; A->data[6][10] = 0.48226079; A->data[6][11] = 0.73182606; A->data[6][12] = 0.3275478 ;

    B->data[0][0] = 0.60469751; B->data[0][1] = 0.88267418; B->data[0][2] = 0.03399421; B->data[0][3] = 0.70252595; B->data[0][4] = 0.50365167; B->data[0][5] = 0.79567525; B->data[0][6] = 0.09844577; B->data[0][7] = 0.74681258; B->data[0][8] = 0.11282681; B->data[0][9] = 0.32250182; B->data[0][10] = 0.34857092;
    B->data[1][0] = 0.53058087; B->data[1][1] = 0.4131175 ; B->data[1][2] = 0.43029027; B->data[1][3] = 0.50045711; B->data[1][4] = 0.70691464; B->data[1][5] = 0.0234768 ; B->data[1][6] = 0.45166739; B->data[1][7] = 0.64254971; B->data[1][8] = 0.2860069 ; B->data[1][9] = 0.50273978; B->data[1][10] = 0.99349672;
    B->data[2][0] = 0.36429902; B->data[2][1] = 0.33108238; B->data[2][2] = 0.10137691; B->data[2][3] = 0.83069144; B->data[2][4] = 0.96434022; B->data[2][5] = 0.15143141; B->data[2][6] = 0.91048467; B->data[2][7] = 0.64365061; B->data[2][8] = 0.97133136; B->data[2][9] = 0.69569789; B->data[2][10] = 0.1623951 ;
    B->data[3][0] = 0.35552452; B->data[3][1] = 0.56355283; B->data[3][2] = 0.96447833; B->data[3][3] = 0.86450632; B->data[3][4] = 0.73677876; B->data[3][5] = 0.66180468; B->data[3][6] = 0.53518887; B->data[3][7] = 0.50398011; B->data[3][8] = 0.40428647; B->data[3][9] = 0.13831744; B->data[3][10] = 0.28139761;
    B->data[4][0] = 0.84909912; B->data[4][1] = 0.34696024; B->data[4][2] = 0.96050184; B->data[4][3] = 0.20167131; B->data[4][4] = 0.77738546; B->data[4][5] = 0.87794767; B->data[4][6] = 0.56471074; B->data[4][7] = 0.71168652; B->data[4][8] = 0.77948542; B->data[4][9] = 0.62879562; B->data[4][10] = 0.70375618;
    B->data[5][0] = 0.68709912; B->data[5][1] = 0.2033942 ; B->data[5][2] = 0.61693483; B->data[5][3] = 0.87100416; B->data[5][4] = 0.54143846; B->data[5][5] = 0.16292727; B->data[5][6] = 0.61519447; B->data[5][7] = 0.02919316; B->data[5][8] = 0.77924423; B->data[5][9] = 0.28912715; B->data[5][10] = 0.68429147;
    B->data[6][0] = 0.64210595; B->data[6][1] = 0.07497808; B->data[6][2] = 0.04789738; B->data[6][3] = 0.55308106; B->data[6][4] = 0.83046182; B->data[6][5] = 0.82719685; B->data[6][6] = 0.81246139; B->data[6][7] = 0.51437718; B->data[6][8] = 0.61099429; B->data[6][9] = 0.22564603; B->data[6][10] = 0.09639494;
    B->data[7][0] = 0.0849274 ; B->data[7][1] = 0.45698971; B->data[7][2] = 0.15597925; B->data[7][3] = 0.20851522; B->data[7][4] = 0.18692654; B->data[7][5] = 0.46989336; B->data[7][6] = 0.27406264; B->data[7][7] = 0.49986653; B->data[7][8] = 0.48885228; B->data[7][9] = 0.14348066; B->data[7][10] = 0.03074643;
    B->data[8][0] = 0.76405808; B->data[8][1] = 0.01323876; B->data[8][2] = 0.56518982; B->data[8][3] = 0.86698702; B->data[8][4] = 0.62376006; B->data[8][5] = 0.23754584; B->data[8][6] = 0.08179041; B->data[8][7] = 0.97154942; B->data[8][8] = 0.54310028; B->data[8][9] = 0.37557141; B->data[8][10] = 0.2682636 ;
    B->data[9][0] = 0.26795246; B->data[9][1] = 0.07967204; B->data[9][2] = 0.0179287 ; B->data[9][3] = 0.12081001; B->data[9][4] = 0.06675936; B->data[9][5] = 0.4851156 ; B->data[9][6] = 0.36368452; B->data[9][7] = 0.91770786; B->data[9][8] = 0.33157281; B->data[9][9] = 0.48131441; B->data[9][10] = 0.82764678;
    B->data[10][0] = 0.16708381; B->data[10][1] = 0.62471521; B->data[10][2] = 0.61353113; B->data[10][3] = 0.7566157 ; B->data[10][4] = 0.70935391; B->data[10][5] = 0.9827929 ; B->data[10][6] = 0.36568655; B->data[10][7] = 0.26288587; B->data[10][8] = 0.61053518; B->data[10][9] = 0.52265032; B->data[10][10] = 0.90535473;
    B->data[11][0] = 0.69374301; B->data[11][1] = 0.80912246; B->data[11][2] = 0.59196418; B->data[11][3] = 0.17427887; B->data[11][4] = 0.81930166; B->data[11][5] = 0.23044397; B->data[11][6] = 0.45547836; B->data[11][7] = 0.03999718; B->data[11][8] = 0.36000216; B->data[11][9] = 0.60965075; B->data[11][10] = 0.33553223;
    B->data[12][0] = 0.87978223; B->data[12][1] = 0.0787401 ; B->data[12][2] = 0.25119156; B->data[12][3] = 0.5873693 ; B->data[12][4] = 0.05513624; B->data[12][5] = 0.62356314; B->data[12][6] = 0.19530057; B->data[12][7] = 0.71577575; B->data[12][8] = 0.85292218; B->data[12][9] = 0.57944889; B->data[12][10] = 0.17238017;

    C->data[0][0] = 2.87935876; C->data[0][1] = 2.12590843; C->data[0][2] = 2.75694277; C->data[0][3] = 3.43187841; C->data[0][4] = 3.49971546; C->data[0][5] = 2.74710891; C->data[0][6] = 2.81064856; C->data[0][7] = 3.23189504; C->data[0][8] = 3.67826723; C->data[0][9] = 2.53787799; C->data[0][10] = 2.45920687;
    C->data[1][0] = 2.70109788; C->data[1][1] = 1.97492439; C->data[1][2] = 2.58992109; C->data[1][3] = 2.67951934; C->data[1][4] = 3.13396894; C->data[1][5] = 2.58264668; C->data[1][6] = 2.05100982; C->data[1][7] = 3.05169235; C->data[1][8] = 2.98051453; C->data[1][9] = 2.4692761 ; C->data[1][10] = 2.45866017;
    C->data[2][0] = 3.21862025; C->data[2][1] = 2.19837013; C->data[2][2] = 2.540988  ; C->data[2][3] = 2.48316025; C->data[2][4] = 3.05011383; C->data[2][5] = 2.43765311; C->data[2][6] = 2.64390776; C->data[2][7] = 3.04902733; C->data[2][8] = 3.24703957; C->data[2][9] = 2.73772091; C->data[2][10] = 3.2431028 ;
    C->data[3][0] = 4.40485849; C->data[3][1] = 2.78138811; C->data[3][2] = 2.99998013; C->data[3][3] = 4.1541314 ; C->data[3][4] = 4.03601096; C->data[3][5] = 3.76835964; C->data[3][6] = 2.97115333; C->data[3][7] = 3.74300712; C->data[3][8] = 3.85084499; C->data[3][9] = 2.74947699; C->data[3][10] = 2.81626687;
    C->data[4][0] = 4.1318553 ; C->data[4][1] = 2.86373085; C->data[4][2] = 3.15058337; C->data[4][3] = 4.5896404 ; C->data[4][4] = 4.0147989 ; C->data[4][5] = 4.22657828; C->data[4][6] = 2.90623908; C->data[4][7] = 4.3632169 ; C->data[4][8] = 4.11645984; C->data[4][9] = 3.05870377; C->data[4][10] = 3.43763387;
    C->data[5][0] = 4.05145116; C->data[5][1] = 2.67856022; C->data[5][2] = 2.82179359; C->data[5][3] = 3.81426254; C->data[5][4] = 4.1641133 ; C->data[5][5] = 3.22806188; C->data[5][6] = 3.09282445; C->data[5][7] = 3.8366267 ; C->data[5][8] = 3.95608851; C->data[5][9] = 3.02337526; C->data[5][10] = 2.85724511;
    C->data[6][0] = 3.83890419; C->data[6][1] = 2.50843657; C->data[6][2] = 2.76312083; C->data[6][3] = 3.60586034; C->data[6][4] = 4.44810986; C->data[6][5] = 3.20636823; C->data[6][6] = 3.16088101; C->data[6][7] = 3.78215606; C->data[6][8] = 3.95644532; C->data[6][9] = 3.21173466; C->data[6][10] = 2.86114049;
    
    gemm(MATRIX_OP_N, MATRIX_OP_N, A, B, D, 1.0, 0.0);
    result = assert_allclose(D, C, atol, rtol);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(D);

    return result;
}

bool test_gemm_ptr() {
    int A_rows = 7,  A_cols = 13;
    int B_rows = 13, B_cols = 11;
    int C_rows = 7,  C_cols = 11;

    double *A = malloc(A_rows * A_cols * sizeof(double));
    double *B = malloc(B_rows * B_cols * sizeof(double));
    double *C = malloc(C_rows * C_cols * sizeof(double));
    double *D = malloc(C_rows * C_cols * sizeof(double));

    bool result = false;

    double atol = 0.0000001;
    double rtol = 0.0000001;

    A[0] = 0.09542479; A[7]  = 0.14712593; A[14] = 0.88998051; A[21] = 0.63738239; A[28] = 0.60919847; A[35] = 0.63385423; A[42] = 0.07044193; A[49] = 0.80001731; A[56] = 0.57447377; A[63] = 0.41220604; A[70] = 0.44199716; A[77] = 0.34032526; A[84] = 0.30402116;
    A[1] = 0.05253944; A[8]  = 0.3099338 ; A[15] = 0.41934822; A[22] = 0.21770562; A[29] = 0.71618029; A[36] = 0.04691743; A[43] = 0.01123448; A[50] = 0.58888321; A[57] = 0.90963758; A[64] = 0.36686079; A[71] = 0.71321206; A[78] = 0.62204195; A[85] = 0.26497803;
    A[2] = 0.18519507; A[9]  = 0.86807473; A[16] = 0.28382064; A[23] = 0.02517163; A[30] = 0.8736909 ; A[37] = 0.82186416; A[44] = 0.02038312; A[51] = 0.76401447; A[58] = 0.03913221; A[65] = 0.76419994; A[72] = 0.10807324; A[79] = 0.7201944 ; A[86] = 0.45104667;
    A[3] = 0.99036225; A[10] = 0.39912427; A[17] = 0.04770133; A[24] = 0.52822626; A[31] = 0.56031853; A[38] = 0.99224872; A[45] = 0.83712788; A[52] = 0.58753871; A[59] = 0.55986886; A[66] = 0.1237992 ; A[73] = 0.15090809; A[80] = 0.65082056; A[87] = 0.8029521 ;
    A[4] = 0.95523478; A[11] = 0.02029638; A[18] = 0.20414547; A[25] = 0.58910318; A[32] = 0.38610546; A[39] = 0.91633644; A[46] = 0.43657481; A[53] = 0.59146723; A[60] = 0.95725668; A[67] = 0.83482456; A[74] = 0.79017214; A[81] = 0.45145973; A[88] = 0.64938558;
    A[5] = 0.93957926; A[12] = 0.06003473; A[19] = 0.79513834; A[26] = 0.13106148; A[33] = 0.93389421; A[40] = 0.9893287 ; A[47] = 0.32802662; A[54] = 0.58543828; A[61] = 0.77108273; A[68] = 0.48034382; A[75] = 0.05005683; A[82] = 0.71144019; A[89] = 0.18448666;
    A[6] = 0.47556832; A[13] = 0.53139942; A[20] = 0.95901975; A[27] = 0.00252595; A[34] = 0.95524412; A[41] = 0.27680841; A[48] = 0.60158878; A[55] = 0.38262372; A[62] = 0.7563717 ; A[69] = 0.16674703; A[76] = 0.48226079; A[83] = 0.73182606; A[90] = 0.3275478 ;

    B[0] = 0.60469751;  B[13] = 0.88267418; B[26] = 0.03399421; B[39] = 0.70252595; B[52] = 0.50365167; B[65] = 0.79567525; B[78] = 0.09844577; B[91] = 0.74681258;  B[104] = 0.11282681; B[117] = 0.32250182; B[130] = 0.34857092;
    B[1] = 0.53058087;  B[14] = 0.4131175 ; B[27] = 0.43029027; B[40] = 0.50045711; B[53] = 0.70691464; B[66] = 0.0234768 ; B[79] = 0.45166739; B[92] = 0.64254971;  B[105] = 0.2860069 ; B[118] = 0.50273978; B[131] = 0.99349672;
    B[2] = 0.36429902;  B[15] = 0.33108238; B[28] = 0.10137691; B[41] = 0.83069144; B[54] = 0.96434022; B[67] = 0.15143141; B[80] = 0.91048467; B[93] = 0.64365061;  B[106] = 0.97133136; B[119] = 0.69569789; B[132] = 0.1623951 ;
    B[3] = 0.35552452;  B[16] = 0.56355283; B[29] = 0.96447833; B[42] = 0.86450632; B[55] = 0.73677876; B[68] = 0.66180468; B[81] = 0.53518887; B[94] = 0.50398011;  B[107] = 0.40428647; B[120] = 0.13831744; B[133] = 0.28139761;
    B[4] = 0.84909912;  B[17] = 0.34696024; B[30] = 0.96050184; B[43] = 0.20167131; B[56] = 0.77738546; B[69] = 0.87794767; B[82] = 0.56471074; B[95] = 0.71168652;  B[108] = 0.77948542; B[121] = 0.62879562; B[134] = 0.70375618;
    B[5] = 0.68709912;  B[18] = 0.2033942 ; B[31] = 0.61693483; B[44] = 0.87100416; B[57] = 0.54143846; B[70] = 0.16292727; B[83] = 0.61519447; B[96] = 0.02919316;  B[109] = 0.77924423; B[122] = 0.28912715; B[135] = 0.68429147;
    B[6] = 0.64210595;  B[19] = 0.07497808; B[32] = 0.04789738; B[45] = 0.55308106; B[58] = 0.83046182; B[71] = 0.82719685; B[84] = 0.81246139; B[97] = 0.51437718;  B[110] = 0.61099429; B[123] = 0.22564603; B[136] = 0.09639494;
    B[7] = 0.0849274 ;  B[20] = 0.45698971; B[33] = 0.15597925; B[46] = 0.20851522; B[59] = 0.18692654; B[72] = 0.46989336; B[85] = 0.27406264; B[98] = 0.49986653;  B[111] = 0.48885228; B[124] = 0.14348066; B[137] = 0.03074643;
    B[8] = 0.76405808;  B[21] = 0.01323876; B[34] = 0.56518982; B[47] = 0.86698702; B[60] = 0.62376006; B[73] = 0.23754584; B[86] = 0.08179041; B[99] = 0.97154942;  B[112] = 0.54310028; B[125] = 0.37557141; B[138] = 0.2682636 ;
    B[9] = 0.26795246;  B[22] = 0.07967204; B[35] = 0.0179287 ; B[48] = 0.12081001; B[61] = 0.06675936; B[74] = 0.4851156 ; B[87] = 0.36368452; B[100] = 0.91770786; B[113] = 0.33157281; B[126] = 0.48131441; B[139] = 0.82764678;
    B[10] = 0.16708381; B[23] = 0.62471521; B[36] = 0.61353113; B[49] = 0.7566157 ; B[62] = 0.70935391; B[75] = 0.9827929 ; B[88] = 0.36568655; B[101] = 0.26288587; B[114] = 0.61053518; B[127] = 0.52265032; B[140] = 0.90535473;
    B[11] = 0.69374301; B[24] = 0.80912246; B[37] = 0.59196418; B[50] = 0.17427887; B[63] = 0.81930166; B[76] = 0.23044397; B[89] = 0.45547836; B[102] = 0.03999718; B[115] = 0.36000216; B[128] = 0.60965075; B[141] = 0.33553223;
    B[12] = 0.87978223; B[25] = 0.0787401 ; B[38] = 0.25119156; B[51] = 0.5873693 ; B[64] = 0.05513624; B[77] = 0.62356314; B[90] = 0.19530057; B[103] = 0.71577575; B[116] = 0.85292218; B[129] = 0.57944889; B[142] = 0.17238017;

    C[0] = 2.87935876; C[7] = 2.12590843;  C[14] = 2.75694277; C[21] = 3.43187841; C[28] = 3.49971546; C[35] = 2.74710891; C[42] = 2.81064856; C[49] = 3.23189504; C[56] = 3.67826723; C[63] = 2.53787799; C[70] = 2.45920687;
    C[1] = 2.70109788; C[8] = 1.97492439;  C[15] = 2.58992109; C[22] = 2.67951934; C[29] = 3.13396894; C[36] = 2.58264668; C[43] = 2.05100982; C[50] = 3.05169235; C[57] = 2.98051453; C[64] = 2.4692761 ; C[71] = 2.45866017;
    C[2] = 3.21862025; C[9] = 2.19837013;  C[16] = 2.540988  ; C[23] = 2.48316025; C[30] = 3.05011383; C[37] = 2.43765311; C[44] = 2.64390776; C[51] = 3.04902733; C[58] = 3.24703957; C[65] = 2.73772091; C[72] = 3.2431028 ;
    C[3] = 4.40485849; C[10] = 2.78138811; C[17] = 2.99998013; C[24] = 4.1541314 ; C[31] = 4.03601096; C[38] = 3.76835964; C[45] = 2.97115333; C[52] = 3.74300712; C[59] = 3.85084499; C[66] = 2.74947699; C[73] = 2.81626687;
    C[4] = 4.1318553 ; C[11] = 2.86373085; C[18] = 3.15058337; C[25] = 4.5896404 ; C[32] = 4.0147989 ; C[39] = 4.22657828; C[46] = 2.90623908; C[53] = 4.3632169 ; C[60] = 4.11645984; C[67] = 3.05870377; C[74] = 3.43763387;
    C[5] = 4.05145116; C[12] = 2.67856022; C[19] = 2.82179359; C[26] = 3.81426254; C[33] = 4.1641133 ; C[40] = 3.22806188; C[47] = 3.09282445; C[54] = 3.8366267 ; C[61] = 3.95608851; C[68] = 3.02337526; C[75] = 2.85724511;
    C[6] = 3.83890419; C[13] = 2.50843657; C[20] = 2.76312083; C[27] = 3.60586034; C[34] = 4.44810986; C[41] = 3.20636823; C[48] = 3.16088101; C[55] = 3.78215606; C[62] = 3.95644532; C[69] = 3.21173466; C[76] = 2.86114049;
    
    gemm_ptr(A, A_rows, A_cols,
             B, B_rows, B_cols,
             D, C_rows, C_cols);

    result = assert_allclose_ptr(D, C_rows, C_cols,
                                 C, C_rows, C_cols,
                                 atol, rtol);

    free(A);
    free(B);
    free(C);
    free(D);

    return result;
}

bool test_gemv() {
    Matrix *A = NULL;
    Matrix *B = NULL;
    Matrix *C = NULL;
    Matrix *D = NULL;
    bool result = false;

    init_matrix(&A, 5, 3);
    init_matrix(&B, 3, 1);
    init_matrix(&C, 5, 1);
    init_matrix(&D, 5, 1);

    double atol = 0.0000001;
    double rtol = 0.0000001;

    A->data[0][0] = 0.49049321; A->data[0][1] = 0.24897268; A->data[0][2] = 0.04795781;
    A->data[1][0] = 0.58035327; A->data[1][1] = 0.51198431; A->data[1][2] = 0.36600846;
    A->data[2][0] = 0.26368691; A->data[2][1] = 0.62015689; A->data[2][2] = 0.42299237;
    A->data[3][0] = 0.13222859; A->data[3][1] = 0.36744077; A->data[3][2] = 0.33433106;
    A->data[4][0] = 0.65285887; A->data[4][1] = 0.45496871; A->data[4][2] = 0.06068799;

    B->data[0][0] = 0.98640454;
    B->data[1][0] = 0.29576062;
    B->data[2][0] = 0.41498498;

    C->data[0][0] = 0.57736281;
    C->data[1][0] = 0.87577591;
    C->data[2][0] = 0.61905543;
    C->data[3][0] = 0.37784775;
    C->data[4][0] = 0.80372939;

    gemv(A, B, D, 1.0, 0.0);
    result = assert_allclose(D, C, atol, rtol);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return result;
}

bool test_dot() {
    Matrix *A = NULL;
    Matrix *B = NULL;
    bool result = false;

    double actual_dot = 0.0;
    double true_dot = 0.6319121054043978;
    double atol = 0.0000001;
    double rtol = 0.0000001;

    init_matrix(&A, 4, 1);
    init_matrix(&B, 4, 1);

    A->data[0][0] = 0.44915435; 
    A->data[1][0] = 0.49957255; 
    A->data[2][0] = 0.02595143; 
    A->data[3][0] = 0.7406385 ;

    B->data[0][0] = 0.64903383; 
    B->data[1][0] = 0.19000438; 
    B->data[2][0] = 0.75401931; 
    B->data[3][0] = 0.30501639;

    dot(A, B, &actual_dot);
    result = assert_close(actual_dot, true_dot, rtol, atol);

    free_matrix(A);
    free_matrix(B);

    return result;
}

bool test_norm() {
    Matrix *A = NULL;
    Matrix *B = NULL;
    bool result = false;

    double true_norm_A = 1.0002655820973592;
    double true_norm_B = 1.057793316050717 ;
    double norm_A = 0.0;
    double norm_B = 0.0;
    double atol = 0.0000001;
    double rtol = 0.0000001;

    init_matrix(&A, 4, 1);
    init_matrix(&B, 4, 1);

    A->data[0][0] = 0.44915435; 
    A->data[1][0] = 0.49957255; 
    A->data[2][0] = 0.02595143; 
    A->data[3][0] = 0.7406385 ;

    B->data[0][0] = 0.64903383; 
    B->data[1][0] = 0.19000438; 
    B->data[2][0] = 0.75401931; 
    B->data[3][0] = 0.30501639;

    norm(A, &norm_A);
    norm(B, &norm_B);

    result = (assert_close(norm_A, true_norm_A, rtol, atol) && assert_close(norm_B, true_norm_B, rtol, atol));

    free_matrix(A);
    free_matrix(B);

    return result;
}

bool test_axpy() {
    Matrix *A = NULL;
    Matrix *B = NULL;
    Matrix *C = NULL;
    Matrix *D = NULL;
    bool result = false;

    double atol = 0.0000001;
    double rtol = 0.0000001;
    double alpha = -2.3;

    init_matrix(&A, 4, 1);
    init_matrix(&B, 4, 1);
    init_matrix(&C, 4, 1);
    init_matrix(&D, 4, 1);

    A->data[0][0] = 0.44915435; 
    A->data[1][0] = 0.49957255; 
    A->data[2][0] = 0.02595143; 
    A->data[3][0] = 0.7406385 ;

    B->data[0][0] = 0.64903383; 
    B->data[1][0] = 0.19000438; 
    B->data[2][0] = 0.75401931; 
    B->data[3][0] = 0.30501639;
    
    C->data[0][0] = -0.38402118; 
    C->data[1][0] = -0.95901249;
    C->data[2][0] =  0.69433103;
    C->data[3][0] = -1.39845216;
    
    axpy(A, B, D, alpha);
    result = assert_allclose(D, C, atol, rtol);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(D);

    return result;
}

bool test_qr(int n_max) {
    bool result = true;
    for (int i = 2; i < n_max; i++) {
        int n = 10*i;
        Matrix *A = NULL;
        Matrix *Q = NULL;
        Matrix *R = NULL;

        double atol = 0.0000001;
        double rtol = 0.0000001;

        init_matrix(&A, 2*n, n);
        init_matrix(&Q, 2*n, 2*n);
        init_matrix(&R, 2*n, n);

        rand_matrix(A);

        clock_t start = clock(), diff;
        qr(A, Q, R);
        diff = clock() - start;
        int msec = diff * 1000 / CLOCKS_PER_SEC;
        gemm(MATRIX_OP_N, MATRIX_OP_N, Q, R, Q, 1.0, 0.0);
        result = result && assert_allclose(Q, A, atol, rtol);

        if (result == true) {
            printf("qr PASS for problem size: (%d, %d) in: %d milliseconds\n", A->rows, A->cols, msec);
        }

        free_matrix(A);
        free_matrix(Q);
        free_matrix(R);
        
    }
    return result;
}

bool test_gpu_qr() {
    bool result = true;
    int n = 3;

    double atol = 0.0000001;
    double rtol = 0.0000001;

    int A_rows = 2*n;
    int A_cols = n;
    int Q_rows = A_rows;
    int Q_cols = A_rows;
    int R_rows = 2*n;
    int R_cols = n;

    double *A = NULL;
    double *Q = NULL;
    double *R = NULL;

    A = calloc(A_rows * A_cols, sizeof(double));
    Q = calloc(Q_rows * Q_cols, sizeof(double));
    R = calloc(R_rows * R_cols, sizeof(double));

    A[0] =  0.521103; A[6]  =  0.251159; A[12]  =  0.448416; 
    A[1] = -0.557204; A[7]  =  0.614949; A[13]  = -0.261701; 
    A[2] =  0.561759; A[8]  =  0.781652; A[14]  = -0.327026; 
    A[3] =  0.461415; A[9]  = -0.611471; A[15] =  0.422656; 
    A[4] =  0.073847; A[10] =  0.190156; A[16] = -0.787745; 
    A[5] = -0.233277; A[11] = -0.650450; A[17] = -0.508967;

    clock_t start = clock(), diff;
    gpu_qr(A, Q, R, A_rows, A_cols);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;

    free(A);
    free(Q);
    free(R);

    return result;
}

bool test_gpu_block_qr_deterministic() {
    bool result = true;
    int n = 4;
    int r = 2;

    double atol = 0.0000001;
    double rtol = 0.0000001;

    int A_rows = 2*n;
    int A_cols = n;
    int Q_rows = A_rows;
    int Q_cols = A_rows;
    int R_rows = 2*n;
    int R_cols = n;

    double *A = NULL;
    double *Q = NULL;
    double *R = NULL;
    double *A_actual = NULL;

    A = calloc(A_rows * A_cols, sizeof(double));
    Q = calloc(Q_rows * Q_cols, sizeof(double));
    R = calloc(R_rows * R_cols, sizeof(double));

    A_actual = calloc(A_rows * A_cols, sizeof(double));

    A[0] = 0.8054398 ; A[8]  = 0.11770048; A[16] = 0.74435746; A[24] = 0.07596747;
    A[1] = 0.47612782; A[9]  = 0.95610043; A[17] = 0.91532087; A[25] = 0.73867671;
    A[2] = 0.43006959; A[10] = 0.61098952; A[18] = 0.2653968 ; A[26] = 0.61539964;
    A[3] = 0.90222967; A[11] = 0.13762961; A[19] = 0.24488956; A[27] = 0.57760962;
    A[4] = 0.08671578; A[12] = 0.33511532; A[20] = 0.13160944; A[28] = 0.7750951 ;
    A[5] = 0.63046399; A[13] = 0.96516845; A[21] = 0.95523958; A[29] = 0.99198526;
    A[6] = 0.34393792; A[14] = 0.18000136; A[22] = 0.95844227; A[30] = 0.39069116;
    A[7] = 0.71946612; A[15] = 0.91549769; A[23] = 0.6170415 ; A[31] = 0.35973015;

    clock_t start = clock(), diff;
    gpu_block_qr(A, Q, R, A_rows, A_cols, r);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;

    gemm_ptr(Q, Q_rows, Q_cols,
             R, R_rows, R_cols,
             A_actual, A_rows, A_cols);

    result = assert_allclose_ptr(A, A_rows, A_cols,
                                 A_actual, A_rows, A_cols,
                                 rtol, atol);

    free(A);
    free(Q);
    free(R);
    free(A_actual);

    return result;
}

bool test_gpu_block_qr_deterministic2() {
    bool result = true;
    int n = 9;
    int r = 3;

    double atol = 0.0000001;
    double rtol = 0.0000001;

    int A_rows = 2*n;
    int A_cols = n;
    int Q_rows = A_rows;
    int Q_cols = A_rows;
    int R_rows = 2*n;
    int R_cols = n;

    double *A = NULL;
    double *Q = NULL;
    double *R = NULL;
    double *A_actual = NULL;

    A = calloc(A_rows * A_cols, sizeof(double));
    Q = calloc(Q_rows * Q_cols, sizeof(double));
    R = calloc(R_rows * R_cols, sizeof(double));
    A_actual = calloc(A_rows * A_cols, sizeof(double));

    readtxt("A_matrix_18_by_9", A, A_rows, A_cols);

    clock_t start = clock(), diff;
    gpu_block_qr(A, Q, R, A_rows, A_cols, r);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;

    gemm_ptr(Q, Q_rows, Q_cols,
             R, R_rows, R_cols,
             A_actual, A_rows, A_cols);

    result = assert_allclose_ptr(A, A_rows, A_cols,
                                 A_actual, A_rows, A_cols,
                                 rtol, atol);
    free(A);
    free(Q);
    free(R);
    free(A_actual);

    return result;
}

bool test_gpu_block_qr_deterministic3() {
    bool result = true;
    int n = 10;
    int r = 5;

    double atol = 0.0000001;
    double rtol = 0.0000001;

    int A_rows = 2*n;
    int A_cols = n;
    int Q_rows = A_rows;
    int Q_cols = A_rows;
    int R_rows = 2*n;
    int R_cols = n;

    double *A = NULL;
    double *Q = NULL;
    double *R = NULL;
    double *A_actual = NULL;

    A = calloc(A_rows * A_cols, sizeof(double));
    Q = calloc(Q_rows * Q_cols, sizeof(double));
    R = calloc(R_rows * R_cols, sizeof(double));
    A_actual = calloc(A_rows * A_cols, sizeof(double));

    readtxt("A_matrix_20_by_10", A, A_rows, A_cols);

    clock_t start = clock(), diff;
    gpu_block_qr(A, Q, R, A_rows, A_cols, r);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;

    gemm_ptr(Q, Q_rows, Q_cols,
             R, R_rows, R_cols,
             A_actual, A_rows, A_cols);

    result = assert_allclose_ptr(A, A_rows, A_cols,
                                 A_actual, A_rows, A_cols,
                                 rtol, atol);

    printf("A = \n");
    print_ptr(A, A_rows, A_cols);
    free(A);
    free(Q);
    free(R);
    free(A_actual);

    return result;
}

bool test_gpu_block_qr_deterministic4() {
    bool result = true;
    int n = 4;
    int r = 4;

    double atol = 0.0000001;
    double rtol = 0.0000001;

    int A_rows = 2*n;
    int A_cols = n;
    int Q_rows = A_rows;
    int Q_cols = A_rows;
    int R_rows = 2*n;
    int R_cols = n;

    double *A = NULL;
    double *Q = NULL;
    double *R = NULL;
    double *A_actual = NULL;

    A = calloc(A_rows * A_cols, sizeof(double));
    Q = calloc(Q_rows * Q_cols, sizeof(double));
    R = calloc(R_rows * R_cols, sizeof(double));
    A_actual = calloc(A_rows * A_cols, sizeof(double));

    readtxt("A_matrix_8_by_4", A, A_rows, A_cols);

    clock_t start = clock(), diff;
    gpu_block_qr(A, Q, R, A_rows, A_cols, r);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;

    gemm_ptr(Q, Q_rows, Q_cols,
             R, R_rows, R_cols,
             A_actual, A_rows, A_cols);

    result = assert_allclose_ptr(A, A_rows, A_cols,
                                 A_actual, A_rows, A_cols,
                                 rtol, atol);
    free(A);
    free(Q);
    free(R);
    free(A_actual);

    return result;
}

bool test_gpu_block_qr(int n_max, double **time_output) {
    bool result = true;
    double atol = 0.00001;
    double rtol = 0.00001;
    clock_t start, end;
    double *times = malloc(n_max * sizeof(double));
    //double time, avg_time;
    float time;
    int r =  64;
    //int r = 256;

    for (int i = 1; i < n_max + 1; i++) {
        int n = i * 64 * 5;
        int A_rows = 2*n;
        int A_cols = n;
        int Q_rows = A_rows;
        int Q_cols = A_rows;
        int R_rows = 2*n;
        int R_cols = n;

        double *A = NULL;
        double *Q = NULL;
        double *R = NULL;
        double *A_actual = NULL;

        A = calloc(A_rows * A_cols, sizeof(double));
        Q = calloc(Q_rows * Q_cols, sizeof(double));
        R = calloc(R_rows * R_cols, sizeof(double));
        A_actual = calloc(A_rows * A_cols, sizeof(double));

        rand_ptr(A, A_rows * A_cols);

        //for (int k = 0; k < 5; k++) {
        time = 0.0;
            start = clock();
            gpu_block_qr(A, Q, R, A_rows, A_cols, r, &time);
            end = clock();
            time = ((double)(end - start)) / CLOCKS_PER_SEC;
            //avg_time += time;

            gemm_ptr(Q, Q_rows, Q_cols,
                     R, R_rows, R_cols,
                     A_actual, A_rows, A_cols);

            result = assert_allclose_ptr(A, A_rows, A_cols,
                                         A_actual, A_rows, A_cols,
                                         rtol, atol);
            if (result == false) {
                double alpha = -1.0;
                double nrm2_diff = 0.0;
                double nrm2_exp = 0.0;
                double rel_err = 0.0;
                axpy_ptr(A_rows * A_cols, A, A_actual, &alpha);
                nrm2_ptr(A_rows * A_cols, A_actual, &nrm2_diff);
                nrm2_ptr(A_rows * A_cols, A, &nrm2_exp);
                rel_err = nrm2_diff / nrm2_exp;

                printf("FAIL FOR MATRIX SIZE = (%d, %d)\n", A_rows, A_cols);
                printf("RELATIVE ERROR = %f\n", rel_err);
                return false;
            }
        //}   

        //times[i - 1] = avg_time / 5.0;
        //printf("Block QR PASS for size (%d, %d) in time = %lf\n", A_rows, A_cols, times[i - 1]);
        printf("Block QR PASS for size (%d, %d) in time = %f\n", A_rows, A_cols, time);

        free(A);
        free(Q);
        free(R);
        free(A_actual);
    }

    *time_output = times;
    return result;
}

int main() {
    if (test_gemm()) {
        printf("test_gemm: Test PASSED\n");
    } else {
        printf("test_gemm: Test FAILED\n");
    }
    if (test_gemm_ptr()) {
        printf("test_gemm_ptr: Test PASSED\n");
    } else {
        printf("test_gemm_ptr: Test FAILED\n");
    }

    if (test_gemv()) {
        printf("test_gemv: Test PASSED\n");
    } else {
        printf("test_gemv: Test FAILED\n");
    }

    if (test_dot()) {
        printf("test_dot: Test PASSED\n");
    } else {
        printf("test_dot: Test FAILED\n");
    }

    if (test_norm()) {
        printf("test_norm: Test PASSED\n");
    } else {
        printf("test_norm: Test FAILED\n");
    }

    if (test_axpy()) {
        printf("test_axpy: Test PASSED\n");
    } else {
        printf("test_axpy: Test FAILED\n");
    }

    if (test_qr(10)) {
        printf("test_qr: Test PASSED\n");
    } else {
        printf("test_qr: Test FAILED\n");
    }

    if (test_gpu_qr()) {
        printf("test_gpu_qr: Test PASSED\n");
    } else {
        printf("test_gpu_qr: Test FAILED\n");
    }

    //if (test_gpu_block_qr_deterministic()) {
    //    printf("test_gpu_block_qr_deterministic: Test PASSED\n");
    //} else {
    //    printf("test_gpu_block_qr_deterministic: Test FAILED\n");
    //}

    //if (test_gpu_block_qr_deterministic2()) {
    //    printf("test_gpu_block_qr_deterministic2: Test PASSED\n");
    //} else {
    //    printf("test_gpu_block_qr_deterministic2: Test FAILED\n");
    //}

    //if (test_gpu_block_qr_deterministic3()) {
    //    printf("test_gpu_block_qr_deterministic3: Test PASSED\n");
    //} else {
    //    printf("test_gpu_block_qr_deterministic3: Test FAILED\n");
    //}

    //if (test_gpu_block_qr_deterministic4()) {
    //    printf("test_gpu_block_qr_deterministic4: Test PASSED\n");
    //} else {
    //    printf("test_gpu_block_qr_deterministic4: Test FAILED\n");
    //}

    double *time_output = NULL;
    if (test_gpu_block_qr(4, &time_output)) {
        printf("test_gpu_block_qr: Test PASSED\n");
    } else {
        printf("test_gpu_block_qr: Test FAILED\n");
    }
    return 0; 
}

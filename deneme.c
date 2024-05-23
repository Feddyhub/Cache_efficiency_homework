/*
ÖDEV GRUBUMUZ
2021280066 Minel ÜSTGEL
2020280014 Murat Ferhat Derya
2022280136 Yılmaz Eray
2020280005 Feyha Müberra Ay 
*/

/* Kullandigimiz bilgisayarlarin islemcileri:
 1-  12th Gen Intel(R) Core(TM) i7-12650H (16CPUs) ~2.3GHZ
 2-  10th Gen Intel(R) Core(TM) i7-10710U (12CPUs) ~1.6GHZ

*/

//Kodu çalıştırabilmek için, AVX2 talimatları içeren bölümlerinin derlenebilmesi için "gcc -mavx2 Deneme.c -o Deneme" komut satırını kullanarak kodumuzu derledik ve derlediğimiz
//kodun exe sini çalıştırdık.

/*
DENEMELERİMİZ SONUCU
 TRESHOLD = 16
Gecen sure: 2.44 seconds
------------------------
 TRESHOLD = 32
Gecen sure: 1.57 seconds
-------------------------
 TRESHOLD = 64
Gecen sure: 1.36 seconds
--------------------------
 TRESHOLD = 128
Gecen sure: 1.26 seconds
------------------------
 TRESHOLD = 256
Gecen sure: 1.28 seconds

Yukarıdaki test sonuçlarından TRESHOLD değeriv 128 için en optimize sonucu vermiştir. 128 değerinden uzaklaştığımızda süre yine uzamıştır.
Test sürecinde matris içindeki sayıları, değerlerin değişip yanıltmaması adına sabit olarak belirledik. 

*/

// GEREKLI KUTUPHANELERIMIZI EKLEDIK

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b)) //MIN DEGER BULAN FONKSIYON
#define MAX(a, b) ((a) > (b) ? (a) : (b)) // MAKS DEGER BULAN FONKSIYON
#define THRESHOLD 256 // RWCF icin kullanilacak esikdegeri 

// Fonksiyon girdilerimiz
void hybridMultiply(int **A, int **B, int **C, int m, int n, int l);
void rowWiseMultiply(int **A, int **B, int **C, int m, int n, int l);
void addMatrix(int **A, int **B, int **Result, int size);
void subMatrix(int **A, int **B, int **Result, int size);
void allocateMatrix(int ***matrix, int rows, int cols);
void freeMatrix(int **matrix, int rows);
void fillMatrix(int **matrix, int rows, int cols);
void printMatrix(int **matrix, int rows, int cols);
void copyMatrix(int **source, int **dest, int startRow, int startCol, int size);
void combineMatrix(int **C11, int **C12, int **C21, int **C22, int **C, int size);
void padMatrix(int **src, int srcRows, int srcCols, int **dst, int dstRows, int dstCols);

int main() {
    // mXn nXl olmak uzere matris carpimi ve  (strassen) icin deger girdilerimiz
    int m = 456, n = 785, l = 293; // Non-multiple of 32 dimensions
	printf("m = %d, n = %d, l = %d\n TRESHOLD = %d\n",m,n,l,THRESHOLD);
    // Pad_size'in boyutunun katlarini duzenledik
    int maxPadded = MAX(MAX(m, n), l);
    maxPadded = pow(2, (int)log2(maxPadded)+1) ;
    int mPadded = maxPadded;
    int nPadded = maxPadded;
    int lPadded = maxPadded;

    //printf("max = %d, m=%d, n=%d, l=%d\n", maxPadded, mPadded, nPadded, lPadded);

    int **A, **B, **C;
    int **APadded, **BPadded, **CPadded;
    // matris icindeki verileri allocate ediyoruz
    allocateMatrix(&A, m, n);
    allocateMatrix(&B, n, l);
    allocateMatrix(&APadded, mPadded, nPadded);
    allocateMatrix(&BPadded, nPadded, lPadded);
    allocateMatrix(&CPadded, mPadded, lPadded);
    // Matris doldurma
    fillMatrix(A, m, n);
    fillMatrix(B, n, l);
    padMatrix(A, m, n, APadded, mPadded, nPadded);
    padMatrix(B, n, l, BPadded, nPadded, lPadded);

    // Algoritmanın efficient'ını ölçmek zaman kıyaslaması yapabilmek adına clock fonksiyon oluşturduk. (bu zaman padding işlemini dahil etmemektedir)
    clock_t start = clock();
    hybridMultiply(APadded, BPadded, CPadded, mPadded, nPadded, lPadded);
    clock_t end = clock();

    printf("Gecen sure: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    printf("Sonuc matrisi C:\n");
    //printMatrix(CPadded, mPadded, lPadded); // Console ekranını daha temiz gösterebilmek adına devredisi bıraktık

    fflush(stdout);
    // Pointerları en sonunda bellek sağlığı açısından serbest bıraktık
    freeMatrix(A, m);
    freeMatrix(B, n);
    freeMatrix(APadded, mPadded);
    freeMatrix(BPadded, nPadded);
    freeMatrix(CPadded, mPadded);

    return 0;
}

// Matrisi 0 matris ile padding (doldurma) islemi uygulayan fonksiyon
void padMatrix(int **src, int srcRows, int srcCols, int **dst, int dstRows, int dstCols) {
    for (int i = 0; i < dstRows; i++) {
        for (int j = 0; j < dstCols; j++) {
            if (i < srcRows && j < srcCols) {
                dst[i][j] = src[i][j];
            } else {
                dst[i][j] = 0;
            }
        }
    }
}

/* ANA FONKSIYONUMUZ Bu fonksiyonumuzda sirasiyla m, n, l degerlerine thresholddan buyuk oldugu surece strassen algoritmasi uyguladik 
ta ki matrisimizin boyutu esik degerimizden daha kucuk parcalara ayirilana kadar. sonrasinda rowWise algoritmasi ile belirli esige kadar 
kuculmus verimizi cok daha hizli isleyebildik */

void hybridMultiply(int **A, int **B, int **C, int m, int n, int l) {
    if (m <= THRESHOLD || n <= THRESHOLD || l <= THRESHOLD) {
        rowWiseMultiply(A, B, C, m, n, l);
        return;
    }

    int midM = m / 2, midN = n / 2, midL = l / 2;

    int **A11, **A12, **A21, **A22;
    int **B11, **B12, **B21, **B22;
    int **C11, **C12, **C21, **C22;
    int **temp1, **temp2;
    int **P1, **P2, **P3, **P4, **P5, **P6, **P7;

    allocateMatrix(&A11, midM, midN);
    allocateMatrix(&A12, midM, midN);
    allocateMatrix(&A21, midM, midN);
    allocateMatrix(&A22, midM, midN);
    allocateMatrix(&B11, midN, midL);
    allocateMatrix(&B12, midN, midL);
    allocateMatrix(&B21, midN, midL);
    allocateMatrix(&B22, midN, midL);
    allocateMatrix(&C11, midM, midL);
    allocateMatrix(&C12, midM, midL);
    allocateMatrix(&C21, midM, midL);
    allocateMatrix(&C22, midM, midL);
    allocateMatrix(&temp1, midM, midN);
    allocateMatrix(&temp2, midN, midL);
    allocateMatrix(&P1, midM, midL);
    allocateMatrix(&P2, midM, midL);
    allocateMatrix(&P3, midM, midL);
    allocateMatrix(&P4, midM, midL);
    allocateMatrix(&P5, midM, midL);
    allocateMatrix(&P6, midM, midL);
    allocateMatrix(&P7, midM, midL);

    copyMatrix(A, A11, 0, 0, midM);
    copyMatrix(A, A12, 0, midN, midM);
    copyMatrix(A, A21, midM, 0, midM);
    copyMatrix(A, A22, midM, midN, midM);
    copyMatrix(B, B11, 0, 0, midN);
    copyMatrix(B, B12, 0, midL, midN);
    copyMatrix(B, B21, midN, 0, midN);
    copyMatrix(B, B22, midN, midL, midN);

    addMatrix(A11, A22, temp1, midM);
    addMatrix(B11, B22, temp2, midL);
    hybridMultiply(temp1, temp2, P1, midM, midN, midL);

    addMatrix(A21, A22, temp1, midM);
    hybridMultiply(temp1, B11, P2, midM, midN, midL);

    subMatrix(B12, B22, temp2, midL);
    hybridMultiply(A11, temp2, P3, midM, midN, midL);

    subMatrix(B21, B11, temp2, midL);
    hybridMultiply(A22, temp2, P4, midM, midN, midL);

    addMatrix(A11, A12, temp1, midM);
    hybridMultiply(temp1, B22, P5, midM, midN, midL);

    subMatrix(A21, A11, temp1, midM);
    addMatrix(B11, B12, temp2, midL);
    hybridMultiply(temp1, temp2, P6, midM, midN, midL);

    subMatrix(A12, A22, temp1, midM);
    addMatrix(B21, B22, temp2, midL);
    hybridMultiply(temp1, temp2, P7, midM, midN, midL);

    addMatrix(P1, P4, temp1, midM);
    addMatrix(temp1, P7, temp2, midM);
    subMatrix(temp2, P5, C11, midM);

    addMatrix(P3, P5, C12, midM);
    addMatrix(P2, P4, C21, midM);
    addMatrix(P1, P3, temp1, midM);
    addMatrix(temp1, P6, temp2, midM);
    subMatrix(temp2, P2, C22, midM);

    combineMatrix(C11, C12, C21, C22, C, midM);

    freeMatrix(A11, midM);
    freeMatrix(A12, midM);
    freeMatrix(A21, midM);
    freeMatrix(A22, midM);
    freeMatrix(B11, midN);
    freeMatrix(B12, midN);
    freeMatrix(B21, midN);
    freeMatrix(B22, midN);
    freeMatrix(C11, midM);
    freeMatrix(C12, midM);
    freeMatrix(C21, midM);
    freeMatrix(C22, midM);
    freeMatrix(temp1, midM);
    freeMatrix(temp2, midN);
    freeMatrix(P1, midM);
    freeMatrix(P2, midM);
    freeMatrix(P3, midM);
    freeMatrix(P4, midM);
    freeMatrix(P5, midM);
    freeMatrix(P6, midM);
    freeMatrix(P7, midM);
}

/* SIMD optımızasyonu içeren RWCF algoritmamız */
void rowWiseMultiply(int **A, int **B, int **C, int m, int n, int l) {
    int i, j, k;
    for (i = 0; i < m; i++) {
        for (j = 0; j < l; j++) {
            C[i][j] = 0;
        }
    }

    for (i = 0; i < m; i++) {
        for (k = 0; k < n; k++) {
            __m256i aLine = _mm256_set1_epi32(A[i][k]);
            for (j = 0; j < l; j += 8) {
                __m256i bLine = _mm256_loadu_si256((__m256i*)&B[k][j]);
                __m256i cLine = _mm256_loadu_si256((__m256i*)&C[i][j]);
                __m256i result = _mm256_add_epi32(cLine, _mm256_mullo_epi32(aLine, bLine));
                _mm256_storeu_si256((__m256i*)&C[i][j], result);
            }
        }
    }
}

void addMatrix(int **A, int **B, int **Result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            Result[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subMatrix(int **A, int **B, int **Result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            Result[i][j] = A[i][j] - B[i][j];
        }
    }
}

void allocateMatrix(int ***matrix, int rows, int cols) {
    *matrix = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        (*matrix)[i] = (int *)malloc(cols * sizeof(int));
    }
}

void freeMatrix(int **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void fillMatrix(int **matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 65534;
        }
    }
}

void printMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void copyMatrix(int **source, int **dest, int startRow, int startCol, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dest[i][j] = source[startRow + i][startCol + j];
        }
    }
}

void combineMatrix(int **C11, int **C12, int **C21, int **C22, int **C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = C11[i][j];
            C[i][j + size] = C12[i][j];
            C[i + size][j] = C21[i][j];
            C[i + size][j + size] = C22[i][j];
        }
    }
}



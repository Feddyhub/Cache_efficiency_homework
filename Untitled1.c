/* Kullandigimiz bilgisayarlarin islemcileri:
 1-  12th Gen Intel(R) Core(TM) i7-12650H (16CPUs) ~2.3GHZ
 2-  10th Gen Intel(R) Core(TM) i7-10710U (12CPUs) ~1.6GHZ

*/

//GEREKLI KUTUPHANELERIMIZI EKLEDIK

#include <stdio.h>
#include <stdlib.h>
//#include <time.h>
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b)) //MIN DEGER BULAN FONKSIYON
#define MAX(a, b) ((a) > (b) ? (a) : (b)) // MAKS DEGER BULAN FONKSIYON
#define THRESHOLD 2 // RWCF icin kullanilacak esikdegeri 
#define PAD_SIZE 32  // (padding_size=matrisimiz kare matris degil ise 32X nin kati seklinde padding yapiyor)

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
    int m = 128, n = 128, l = 128; // Non-multiple of 32 dimensions

 //Pad_size'in boyutunun katlarini duzenledik
  
//    int mPadded = ceil((m + PAD_SIZE - 1) / 32) * 32;
//    int nPadded = ceil((n + PAD_SIZE - 1) / 32) * 32;
//    int lPadded = ceil((l + PAD_SIZE - 1) / 32) * 32;
	int maxPadded = MAX(MAX(m, n), l);
	maxPadded = round(((float)(m + PAD_SIZE - 1) / PAD_SIZE)) * PAD_SIZE;
	int mPadded = maxPadded;
    int nPadded = maxPadded;
    int lPadded = maxPadded;

    //printf("m=%d, n=%d, l=%d\n",mPadded,nPadded,lPadded);

    int **A, **B, **C;
    int **APadded, **BPadded, **CPadded;
    // matris icindeki verileri allocate ediyoruz
    allocateMatrix(&A, m, n);
    allocateMatrix(&B, n, l);
    allocateMatrix(&APadded, mPadded, nPadded);
    allocateMatrix(&BPadded, nPadded, lPadded);
    allocateMatrix(&CPadded, mPadded, lPadded);
	//Matris doldurma
    fillMatrix(A, m, n);
    fillMatrix(B, n, l);
    padMatrix(A, m, n, APadded, mPadded, nPadded);
    padMatrix(B, n, l, BPadded, nPadded, lPadded);


	/* algoritmanin efficient'ini olcmek zaman kiyaslamasi 
	yapabilmek adina  clock fonksiyon olusturduk.( bu zaman padding islemini dahil etmemektedir)
	Guncelleme clock islemi hizi azalttigi icin yorum satirina cevrilmistir 
    clock_t start = clock();
    hybridMultiply(APadded, BPadded, CPadded, mPadded, nPadded, lPadded);
    clock_t end = clock();*/

    //printf("Gecen sure: %.2f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    //printf("Sonuc matrisi C:\n");
    //printMatrix(CPadded, mPadded, lPadded); console ekranini daha temiz gosterebilmek adina devredisi biraktik


    //fflush(stdout);
    //pointerlari en sonunda bellek sagligi acisindan serbest biraktik
    freeMatrix(A, m);
    freeMatrix(B, n);
    freeMatrix(APadded, mPadded);
    freeMatrix(BPadded, nPadded);
    freeMatrix(CPadded, mPadded);

    return 0;
}

//matrisi 0 matris ile padding (doldurma) islemi uygulayan fonksiyon
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
/* ANA FONKSIYONUMUZ Bu fonksiyonumuzda sirasiyla m ,n,l 
degerlerine thresholddan buyuk oldugu surece strassen algoritmasi uyguladik 
ta ki matrisimizin boyutu esik degerimizden daha kucuk parcalara ayirilana kadar. sonrasinda rowWise algoritmasi ile
belirli esige kadar kuculmus verimizi cok daha hizli isleyebildik*/

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
    subMatrix(temp1, P5, temp2, midM);
    addMatrix(temp2, P7, C11, midM);

    addMatrix(P3, P5, C12, midM);

    addMatrix(P2, P4, C21, midM);

    addMatrix(P1, P3, temp1, midM);
    subMatrix(temp1, P2, temp2, midM);
    addMatrix(temp2, P6, C22, midM);

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
    freeMatrix(temp2, midM);
    freeMatrix(P1, midM);
    freeMatrix(P2, midM);
    freeMatrix(P3, midM);
    freeMatrix(P4, midM);
    freeMatrix(P5, midM);
    freeMatrix(P6, midM);
    freeMatrix(P7, midM);
}


//rowWiseMultiply fonksiyonumuz (tek fonksiyon)
void rowWiseMultiply(int **A, int **B, int **C, int m, int n, int l) {
    int mPadded = (m + PAD_SIZE - 1) / PAD_SIZE * PAD_SIZE;
    int nPadded = (n + PAD_SIZE - 1) / PAD_SIZE * PAD_SIZE;
    int lPadded = (l + PAD_SIZE - 1) / PAD_SIZE * PAD_SIZE;

    int **APadded, **BPadded;
    allocateMatrix(&APadded, mPadded, nPadded);
    allocateMatrix(&BPadded, nPadded, lPadded);

    padMatrix(A, m, n, APadded, mPadded, nPadded);
    padMatrix(B, n, l, BPadded, nPadded, lPadded);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < l; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    freeMatrix(APadded, mPadded);
    freeMatrix(BPadded, nPadded);
}
//Strassen algoritmasi icin gereken operator fonksiyonumuz (1)
void addMatrix(int **A, int **B, int **Result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            Result[i][j] = A[i][j] + B[i][j];
        }
    }
}
//Strassen algoritmasi icin gereken operator fonksiyonumuz (2)
void subMatrix(int **A, int **B, int **Result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            Result[i][j] = A[i][j] - B[i][j];
        }
    }
}

#include <stdalign.h>
//bizim kullandigimiz malloc
void allocateMatrix(int ***matrix, int rows, int cols) {

    matrix = (int *)malloc(rows * sizeof(int *));
    if (*matrix == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix rows\n");
        exit(EXIT_FAILURE);
    }
    


    for (int i = 0; i < rows; i++) {
        (*matrix)[i] = (int *)malloc(cols * sizeof(int));
        if ((*matrix)[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for matrix cols at row %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

/*Linux Unix tabanli isletim sistemlerinde calisabilen kod

void allocateMatrix(int ***matrix, int rows, int cols) {
    matrix = (int *)malloc(rows * sizeof(int *));
    if (*matrix == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix rows\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < rows; i++) {
        if (posix_memalign((void **)&((*matrix)[i]), 32, cols * sizeof(int)) != 0) {
            fprintf(stderr, "Memory allocation failed for matrix cols at row %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
}



*/





//Kullandiktan sonra isimiz biten bellekteki yeri bosaltan kod
void freeMatrix(int **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void fillMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 1; // Rastgele degerler girdik        DEGISTIR
        }
    }
}
//Ciktilarimizi alabilmek icin clean bir print fonksiyonumuz
void printMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf(" %d ", matrix[i][j]);
        }
        printf("\n");
    }
}

//Strassen algoritmasi icin gereken operator fonksiyonumuz (3)
void copyMatrix(int **source, int **dest, int startRow, int startCol, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dest[i][j] = source[startRow + i][startCol + j];
        }
    }
}
//Strassen algoritmasi icin gereken operator fonksiyonumuz (4)
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
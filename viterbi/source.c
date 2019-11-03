#include "dnn_hmm.h"
#include "dnn_w1.h"
//#include "hmm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <io.h>
#include <conio.h>
#include <math.h>

#define MAX_PRO 6 //maximum number of pronunciation in one word
#define N_WORD 13 //number of word
#define N_STATE_WORD ((MAX_PRO) * (N_STATE)) //number of state in one wordhmm
#define N_STATE_TOTAL ((N_WORD) * (N_STATE_WORD)) //number of state in utterance hmm
#define MAX_T 570
#define LAMBDA 9
#define CUT 3
// two is inserted back and forth in every test case so i cut it off
// i think that is the sound of the button or some other noise sounds like two

struct _finddata_t fd;

typedef struct{
	char *name;
	int num_of_state;
	float tp[N_STATE_WORD+2][N_STATE_WORD+2];
	int state[N_STATE_WORD];
}wordHmm;

typedef struct{
	char *name;
	hmmType pro[MAX_PRO];
}wordInfo;

wordHmm * make_wordHmm(wordInfo *info);
hmmType pro(char *name);
int pro_index(char *name);
float log_prob(int state, float input[]);

int isFileOrDir();
void FileSearch(char file_path[]);

float h[N_LAYER - 1][N_HIDDEN]; // input, hidden layer output
float out[N_OUT]; // output layer output

void cal_out(float input[]);

int main()
{
	float bi[N_WORD][N_WORD] = {
	//sil		eight		five		four		nine		oh			one			seven		six			three		two			zero		zero2
	{0,			0.093813,	0.101901,	0.081278,	0.082491,	0.092196,	0.095431,	0.08977,	0.091791,	0.082895,	0.103518,	0.042458,	0.042458},
	{0.143561,	0.075169,	0.074553,	0.080715,	0.083795,	0.073937,	0.081331,	0.07825,	0.071473,	0.075169,	0.074553,	0.043746,	0.043746},
	{0.149581,	0.084462,	0.072856,	0.063185,	0.068988,	0.075435,	0.088330,	0.074146,	0.075435,	0.082527,	0.084462,	0.040296,	0.040296},
	{0.137686,	0.084034,	0.077569,	0.083387,	0.065934,	0.078862,	0.071105,	0.083387,	0.080802,	0.087266,	0.075630,	0.037168,	0.037168},
	{0.160750,	0.077696,	0.070998,	0.081715,	0.069658,	0.077026,	0.073007,	0.087073,	0.079705,	0.077026,	0.073677,	0.035834,	0.035834},
	{0.142677,	0.075126,	0.079545,	0.080808,	0.077652,	0.157828,	0.086490,	0.077652,	0.075758,	0.080808,	0.065657,	0,			0},
	{0.149686,	0.082390,	0.073585,	0.073585,	0.069182,	0.072327,	0.074214,	0.066667,	0.083019,	0.080503,	0.093082,	0.040880,	0.040880},
	{0.137573,	0.086308,	0.077223,	0.076574,	0.079169,	0.081116,	0.074627,	0.080467,	0.081765,	0.077223,	0.081116,	0.033420,	0.033420},
	{0.148447,	0.079503,	0.081988,	0.083230,	0.073292,	0.077019,	0.079503,	0.075155,	0.078882,	0.072671,	0.068944,	0.040683,	0.040683},
	{0.132063,	0.079365,	0.078730,	0.069206,	0.078095,	0.085079,	0.083810,	0.071746,	0.086349,	0.067937,	0.086349,	0.040635,	0.040635},
	{0.138614,	0.081683,	0.066832,	0.080446,	0.074257,	0.082921,	0.078589,	0.076733,	0.081683,	0.081064,	0.079827,	0.038675,	0.038675},
	{0.131562,	0.078431,	0.071474,	0.082226,	0.078431,	0,			0.068944,	0.067679,	0.084124,	0.088552,	0.080961,	0.083807,	0.083807}};
	
	float uni[N_WORD] = {0.990000, 0.000938, 0.000896, 0.000894, 0.000862, 0.000915, 0.000918, 0.000890, 0.000930, 0.000910, 0.000934, 0.000456, 0.000456};

	wordInfo info[N_WORD] = {
	{"sil", {pro("sil")}},
	{"eight", {pro("ey"), pro("t"), pro("sp")}},
	{"five", {pro("f"), pro("ay"), pro("v"), pro("sp")}},
	{"four", {pro("f"), pro("ao"), pro("r"), pro("sp")}},
	{"nine", {pro("n"), pro("ay"), pro("n"), pro("sp")}},
	{"oh", {pro("ow"), pro("sp")}},
	{"one", {pro("w"), pro("ah"), pro("n"), pro("sp")}},
	{"seven", {pro("s"), pro("eh"), pro("v"), pro("ah"), pro("n"), pro("sp")}},
	{"six", {pro("s"), pro("ih"), pro("k"), pro("s"), pro("sp")}},
	{"three", {pro("th"), pro("r"), pro("iy"), pro("sp")}},
	{"two", {pro("t"), pro("uw"), pro("sp")}},
	{"zero", {pro("z"), pro("ih"), pro("r"), pro("ow"), pro("sp")}},
	{"zero", {pro("z"), pro("iy"), pro("r"), pro("ow"), pro("sp")}}};
	
	//make wordhmm
	wordHmm *word[N_WORD];
	
	int i = 0;
	for (i = 0 ;i < N_WORD; i++){
		word[i] = make_wordHmm(&(info[i]));
	}
	
	//make utterance hmm
	float tp[N_STATE_TOTAL+2][N_STATE_TOTAL+2] = {0};
	int state[N_STATE_TOTAL] = {0};
	
	int n = 0, m = 0;
	for(i = 0; i < N_WORD; i++){
		tp[0][i*N_STATE_WORD +1] = uni[i];
		for(n = 0; n < word[i]->num_of_state; n++){
			state[i*N_STATE_WORD + n] = word[i]->state[n];
			for(m = 0; m < word[i]->num_of_state; m++){
				tp[i*N_STATE_WORD + n + 1][i*N_STATE_WORD + m + 1] = word[i]->tp[n + 1][m + 1];
			}
		}
		for(n = 0; n < N_WORD; n++){
			tp[i*N_STATE_WORD + word[i]->num_of_state][n*N_STATE_WORD + 1] =
			tp[i*N_STATE_WORD + word[i]->num_of_state][n*N_STATE_WORD + 1] +
			bi[i][n] * word[i]->tp[word[i]->num_of_state][word[i]->num_of_state + 1];

			tp[i*N_STATE_WORD + word[i]->num_of_state - 1][n*N_STATE_WORD + 1] =
			tp[i*N_STATE_WORD + word[i]->num_of_state - 1][n*N_STATE_WORD + 1] +
			bi[i][n] * word[i]->tp[word[i]->num_of_state - 1][word[i]->num_of_state + 1];
		}
	}

	//make file which have all test file's route
	char file_path[_MAX_PATH] = "tst";
	FILE* route = fopen("route.txt","w");
	
	FileSearch(file_path);
	fclose(route);

	//make rec file
	FILE* rec = fopen("recognized.txt","w");
	fprintf(rec,"#!MLF!#\n");
	fclose(rec);

	//viterbi alg.
	route = fopen("route.txt","r");
	rec = fopen("recognized.txt","a");
	char test_route[_MAX_PATH];
	while(fgets(test_route,_MAX_PATH,route)){
		test_route[strlen(test_route) - 1] = '\0';
		
		FILE *test_file = fopen(test_route,"r");
		int T = 0, t = 0, d = 0, i = 0, j = 0;
		float compare = 0;
		float input[N_DIMENSION] = {0};
		float disc[MAX_T][N_STATE_TOTAL] = {0};	// discreminant func : log (delta t(i))
		int pre_state[MAX_T][N_STATE_TOTAL] = {0};

		fscanf(test_file,"%d",&T);
		T = T - 2 * CUT; //cut back and forth
		fscanf(test_file,"%d",&d); // drop 39

		//initialization
		t = 0;
		for(i = 0; i < CUT + 1; i++){
			for(d = 0; d < N_DIMENSION; d++){
				fscanf(test_file,"%f",&(input[d]));
			}
		}
		cal_out(input);

		for(i = 0; i < N_STATE_TOTAL; i++){
			if(tp[t][i+1]){
			disc[t][i] = LAMBDA * log(tp[t][i+1]) + log(out[state[i]]);
			pre_state[t][i] = -1;
			}
		}
		
		//recursion
		for(t = 1; t < T; t++){
			//read input
			for(d = 0; d < N_DIMENSION; d++){
				fscanf(test_file,"%f",&(input[d]));
			}
			cal_out(input);

			//max, argmax
			for(j = 0; j < N_STATE_TOTAL; j++){
				//delta t(j) = max(i) (delta t-1(i) * a ij) * b j(input)
				i = 0;
				while(((disc[t-1][i] == 0)||(tp[i+1][j+1] == 0))&&(i < N_STATE_TOTAL)) i++;
				if(i == N_STATE_TOTAL) continue;
				
				disc[t][j] = disc[t-1][i] + LAMBDA * log(tp[i+1][j+1]);
				pre_state[t][j] = i;
	
				i++;				
				for(; i < N_STATE_TOTAL; i++){
					if(!((disc[t-1][i] == 0)||(tp[i+1][j+1] == 0))){
						compare = disc[t-1][i] + LAMBDA * log(tp[i+1][j+1]);
						if(compare > disc[t][j]){
							disc[t][j] = compare;
							pre_state[t][j] = i;
						}
					}
				}
				disc[t][j] = disc[t][j] + log(out[state[i]]);
			}			
		}

		//find last state		
		int q[MAX_T] = {};
		compare = disc[T-1][0];
		q[T-1] = 0;
		for(i = 1; i < N_STATE_TOTAL; i++){
			if(disc[T-1][i] == 0) continue;
			if(compare < disc[T-1][i]){
				compare = disc[T-1][i];
				q[T-1] = i;
			}
		}

		//backtracking
		for(t = T - 2; t >= 0; t--){
			q[t] = pre_state[t+1][q[t+1]];
		}
		
		//record
		test_route[strlen(test_route)-4] = '\0'; //drop .txt
		fprintf(rec,"\"%s.rec\"\n",test_route);
		
		for(t = 0; t < T - 1; t++){
			if(q[t]%N_STATE_WORD == 0){
				if((q[t] != q[t+1])&&(q[t] != 0)){
				fprintf(rec,"%s\n",word[q[t]/N_STATE_WORD]->name);
				}
			}
		}
		if((q[t]%18 == 0)&&(q[t] != 0)){
			fprintf(rec,"%s\n",word[q[t]/N_STATE_WORD]->name);
		}
		
		fprintf(rec,".\n");
		printf("%s done\n",test_route);

		fclose(test_file);
//		break;
	}
	fclose(rec);
	fclose(route);
	return 0;
}

// ======== main end ========

hmmType pro(char *name){
	int i;
	for(i = 0; i < sizeof(phones)/sizeof(hmmType); i++){
		if(!strcmp(name,phones[i].name)){
			return phones[i];
		}
	}
}
int pro_index(char *name){
	int i;
	for(i = 0; i < sizeof(phones)/sizeof(hmmType); i++){
		if(!strcmp(name,phones[i].name)){
			return i;
		}
	}
}
wordHmm *make_wordHmm(wordInfo *info){
	wordHmm *r;
	r = (wordHmm *)malloc(sizeof(wordHmm));
	r->name = info->name;
	r->num_of_state = 0;
	int i, n, m;
	r->tp[0][1] = 1;
	for(i = 0; i < MAX_PRO; i++){
		if(!info->pro[i].name) ;
		else if(!strcmp(info->pro[i].name,"sp")){
			r->state[i*N_STATE] = pro_index(info->pro[i].name) * 3;
			r->tp[i*N_STATE][i*N_STATE+1] = info->pro[i-1].tp[N_STATE][N_STATE + 1] * info->pro[i].tp[0][1];
			r->tp[i*N_STATE][i*N_STATE+2] = info->pro[i-1].tp[N_STATE][N_STATE + 1] * info->pro[i].tp[0][2];
			r->tp[i*N_STATE+1][i*N_STATE+1] = info->pro[i].tp[1][1];
			r->tp[i*N_STATE+1][i*N_STATE+2] = info->pro[i].tp[1][2];
			r->num_of_state = r->num_of_state + 1;
		}
		else{
			for(n = 0; n < N_STATE; n++){
				r->state[i*N_STATE + n] = pro_index(info->pro[i].name) * 3 + n;
				for(m = 0; m < N_STATE; m++){
					r->tp[i*N_STATE + n + 1][i*N_STATE + m + 1] = info->pro[i].tp[n + 1][m + 1];
				}
			}
			r->tp[(i+1)*N_STATE][(i+1)*N_STATE + 1] = info->pro[i].tp[N_STATE][N_STATE + 1];
			r->num_of_state = r->num_of_state + N_STATE;			
		}
	}
	return r;
}

float log_N_dis(float input[], float mean[], float var[]){
	double pro = -log(2.0*M_PI)*N_DIMENSION/2.0;
	int n = 0;
	for(n = 0; n < N_DIMENSION; n++){
		pro = pro - log(var[n])/2.0 - ((1.0/2.0)*(input[n]-mean[n])*(input[n]-mean[n]))/var[n];
	}
	return pro;
}

int isFileOrDir(){
    if (fd.attrib & _A_SUBDIR)
        return 0;
    else
        return 1;
 
}
void FileSearch(char file_path[]){
	FILE* route = fopen("route.txt","a");
    intptr_t handle;
    int check = 0;
    char file_path2[_MAX_PATH];
	
    strcat(file_path, "\\");
    strcpy(file_path2, file_path);
    strcat(file_path, "*");
 
    if ((handle = _findfirst(file_path, &fd)) == -1){
        printf("No such file or directory\n");
        return;
    }
 
    while (_findnext(handle, &fd) == 0){
        char file_pt[_MAX_PATH];
        strcpy(file_pt, file_path2);
        strcat(file_pt, fd.name);
 
        check = isFileOrDir();
 
        if (check == 0 && fd.name[0] != '.'){
            FileSearch(file_pt);
        }
        else if (check == 1 && fd.size != 0 && fd.name[0] != '.'){
            fprintf(route,"%s\n", file_pt);
        }
    }
    _findclose(handle);
    fclose(route);
}

void cal_out(float input[]){	
	int l = 0,T = 0, t = 0, d = 0, i = 0, j = 0, k = 0;
	float sum = 0;
	// input layer output
	for(i = 0; i < N_HIDDEN; i++){
		h[0][i] = in_w[i][0];
		for(d = 0; d < N_DIMENSION; d++){
			h[0][i] += (in_w[i][d+1] * input[d]);
		}
		h[0][i] = 1.0 / (1.0 + exp(h[0][i]));
	}
	// hidden layer output
	for(l = 0; l < N_LAYER - 2 ; l++){
		for(i = 0; i < N_HIDDEN; i++){
			h[l+1][i] = hidden_w[l][i][0];
			for(j = 0; j < N_HIDDEN; j++){
				h[l+1][i] += (hidden_w[l][i][j+1] * h[l][j]);
			}
			h[l+1][i] = 1.0 / (1.0 + exp(h[l+1][i]));
		}
	}
	// output layer output
	sum = 0;
	for(i = 0; i < N_OUT; i++){
		out[i] = out_w[i][0];
		for(j = 0; j < N_HIDDEN; j++){
			out[i] += (out_w[i][j+1] * h[N_LAYER - 2][j]);
		}
		sum += out[i];
	}
	for(i = 0; i < N_OUT; i++){
		out[i] = out[i] / sum;
	}
}

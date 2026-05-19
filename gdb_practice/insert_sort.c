/*
  программа сортировки вставками с несколькими ошибками
  
  пример компиляции
  gcc -g prog_b318.c 
  
  программе через командную строку передаются целые числа,
  которые нужно отсортировать  (максимально 100 чисел)
  запуск (linux)
  ./a.out 10 5 11 2 3
  
  запуск (win)
  ./a.exe 10 5 11 2 3
  
  программа печатает сообщение о запуске
  и зависает
*/
#include <stdio.h>
#include <stdlib.h>

int x[100], y[100], n_in, num_y = 0;

void get_args(int xx, char **y){
	int i;
	n_in = xx - 1;
	for (i = 0; i < n_in; i++)
		x[i] = atoi(y[i+1]);
}
void move(int jj){ 
	int k;
	for (k = num_y; k > jj; k--)
		y[k] = y[k-1];
}
void insert(int new_y){ 
	int j;
	if (num_y == 0) { 
		y[0] = new_y; num_y++;
		return;
	}
	for (j = 0; j < num_y; j++) {
		if (new_y < y[j]) {
			move(j);
			y[j] = new_y; num_y++;
			return;
		}
	}
	y[num_y]=new_y;
	num_y++;
}

void process_data()
{
	for (int i = 0; i < n_in; i++)
		insert(x[i]);
}

void show_res(){
	int i;
	for (i = 0; i < n_in; i++)
		printf("%d ",y[i]);
	printf(" \n");
	}

int main(int argc, char ** argv)
{
    printf("program start\n");
	get_args(argc, argv);
	process_data();
	show_res();
	printf("program end\n");
	exit(0);
}

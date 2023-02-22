#include "./SumArray/sumarray.cuh"
#include "./SumArrayWithTiming/sumarraywithtiming.cuh"
#include "./memTransferCuda/memtransfer.cuh"

int main()
{
	printf("-----MEMORY TRANSFER EXAMPLE----\n");
	memtransferexample();
	printf("-----SUM ARRAY EXAMPLE----\n");
	sumarrayexample();
	printf("-----SUM ARRAY WITH TIMING EXAMPLE----\n");
	sumarraywithtimingexample();
	return 0;
}

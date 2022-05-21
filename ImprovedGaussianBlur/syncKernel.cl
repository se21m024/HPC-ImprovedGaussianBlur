

__kernel void sync_demo_kernel(
	__global const int* A,
	__global int* B
)
{

	__local int localTemp[4];
	size_t localId = get_local_id(0);
	size_t localSize = get_local_size(0);


	//Every thread reads its value and
	// first stores it inot local memory
	int result = A[localId];
	localTemp[localId] = result;
	//mem_fence(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE);

	// and add own value to neighboring
	// threads value stored in local memory
	size_t neighbor = localId + 1;
	if (neighbor < localSize) {
		result += localTemp[neighbor];
	}

	printf("RESULT: %u LOCAL_ID: %u \n ", result, localId);

	B[localId] = result;
}
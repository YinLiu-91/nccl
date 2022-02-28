CC=nvcc 
# use 'w' to ignore 'warning: pointer of type ‘void *’ used in arithmetic [-Wpointer-arith]'
CFLAGS+= -g -G -w
LIBS= -lnccl -lmpi

all: example_3.out example_3alltoall.out example_3gather.out example_3scatter.out

OBJS=example_3
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS}

OBJS=example_3alltoall
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS=example_3gather
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS=example_3scatter
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

clean: 
	${RM} *.out -r
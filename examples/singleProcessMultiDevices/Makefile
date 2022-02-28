CC=nvcc 
# use 'w' to ignore 'warning: pointer of type ‘void *’ used in arithmetic [-Wpointer-arith]'
CFLAGS+= -g -G -w
LIBS= -lnccl -lmpi

all: example_1.out example_1reduceScatter.out example_1sendRecv.out example_1sendRecvbind.out 

OBJS=example_1
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS}

OBJS=example_1reduceScatter
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS=example_1sendRecv
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS= example_1sendRecvbind
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

clean: 
	${RM} *.out -r
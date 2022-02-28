CC=nvcc 
# use 'w' to ignore 'warning: pointer of type ‘void *’ used in arithmetic [-Wpointer-arith]'
CFLAGS+= -g -G -w
LIBS= -lnccl -lmpi

all: example_2allreduce.out example_2brdcast.out example_2gather.out example_2reduce.out example_2reduceScatter.out \
example_2sendRecv.out  example_2sendRecvbind.out

OBJS=example_2allreduce
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS}

OBJS=example_2brdcast
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS=example_2gather
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS= example_2reduce
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS= example_2reduceScatter
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS= example_2sendRecv
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

OBJS= example_2sendRecvbind
${OBJS}.out: ${OBJS}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

clean: 
	${RM} *.out -r
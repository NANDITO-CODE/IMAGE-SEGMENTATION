INCDIR = -I.
DBG    = -g
OPT    = -O3
CPP    = g++
MPICPP = mpic++
CFLAGS = $(DBG) $(OPT) $(INCDIR)
LINK   = -lm 

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: segment segment-mpi

segment: segment.cpp segment-image.h segment-graph.h disjoint-set.h
	$(CPP) $(CFLAGS) -o segment segment.cpp $(LINK)

segment-mpi: segment-mpi.cpp segment-image-mpi.h segment-graph.h disjoint-set.h
	$(MPICPP) $(CFLAGS) -o segment-mpi segment-mpi.cpp $(LINK)

clean:
	/bin/rm -f segment segment-mpi *.o

clean-all: clean
	/bin/rm -f *~




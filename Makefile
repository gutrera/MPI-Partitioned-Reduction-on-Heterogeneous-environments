MPICC= mpicxx

testPR: testPR.cu
	$(MPICC) testPR.cu -o testPR

clean: 
	rm testPR

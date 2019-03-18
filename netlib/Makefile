include ../Makefile.rule

BLAS_NETLIB = blas-3.8.0
BLAS_DIR = blas

obj: | $(BLAS_DIR)
	@echo "compiling blas ..."
	(cd $(BLAS_DIR); $(MAKE) all)

$(BLAS_DIR):
	@echo "downloading blas ..."
	wget www.netlib.org/blas/$(BLAS_NETLIB).tgz -O $(BLAS_NETLIB).tgz
	mkdir -p $(BLAS_DIR)
	tar -xzf $(BLAS_NETLIB).tgz -C $(BLAS_DIR) --strip-components 1

clean:
ifneq ($(wildcard $(BLAS_DIR)), )
	make -C $(BLAS_DIR) clean;
endif
#	if [ -d "./$(BLAS_DIR)" ]; then \
#		make -C $(BLAS_DIR) clean; \
#	fi

deep_clean:
	rm -rf $(BLAS_DIR)
	rm -rf $(BLAS_NETLIB).tgz

.PHONY: all deep_clean

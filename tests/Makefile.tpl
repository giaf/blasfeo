# ----------- Include


TESTS_DIR=$(BLASFEO_DIR)/tests
ABS_BINARY_DIR=$(TESTS_DIR)/$(BINARY_DIR)

include $(BLASFEO_DIR)/Makefile.rule
# ----------- Envs


LIBS =
SHARED_LIBS =

LIBS += $(ABS_BINARY_DIR)/libblasfeo_ref.a
SHARED_LIBS += -Wl,-rpath=$(ABS_BINARY_DIR) -L $(ABS_BINARY_DIR) -lblasfeo_ref

LIBS += $(ABS_BINARY_DIR)/libblasfeo.a
SHARED_LIBS += -Wl,-rpath=$(ABS_BINARY_DIR) -L $(ABS_BINARY_DIR) -lblasfeo

include $(BLASFEO_DIR)/Makefile.blas

{% for flag, value in cflags.items() %}
{% if value %}CFLAGS += -D{{flag | upper}}={{value}}{% else %}CFLAGS += -D{{flag | upper}}{% endif %}
{% endfor %}

{% if TEST_BLAS_API in cflags %}
ifeq ($(REF_BLAS), 0)
$(error No REF_BLAS specified, install specify one reference blas implementation i.e. OPENBLAS)
{% endif %}

test.o:
	# build executable obj $(ABS_BINARY_DIR)
	$(CC) $(CFLAGS) -c $(TESTS_DIR)/test.c -o $(ABS_BINARY_DIR)/test.o
	$(CC) $(CFLAGS) $(ABS_BINARY_DIR)/test.o -o $(ABS_BINARY_DIR)/test.out $(LIBS)
	$(ABS_BINARY_DIR)/test.out


update_lib:
	mkdir -p $(ABS_BINARY_DIR)/
	cp ../lib/libblasfeo.a ./$(ABS_BINARY_DIR)
	cp ../lib/libblasfeo_ref.a ./$(ABS_BINARY_DIR)

run: test.o

update: run
full: update_lib run

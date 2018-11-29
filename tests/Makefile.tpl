# ----------- Include
include ../Makefile.rule

# ----------- Envs

LIBS =
SHARED_LIBS =

LIBS += $(BINARY_DIR)/libblasfeo_ref.a
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo_ref

LIBS += $(BINARY_DIR)/libblasfeo.a
SHARED_LIBS += -Wl,-rpath=$(BINARY_DIR) -L $(BINARY_DIR) -lblasfeo

include ../Makefile.blas

{% for flag, value in cflags.items() %}
{% if value %} CFLAGS += -D{{flag | upper}}={{value}} {% else %} CFLAGS += -D{{flag | upper}} {% endif %} {% endfor %}

{% if TEST_BLAS_API in cflags %}
ifeq ($(REF_BLAS), 0)
$(error No REF_BLAS specified, install specify one reference blas implementation i.e. OPENBLAS)
{% endif %}

test.o: test.c
	# build executable obj $(BINARY_DIR)
	$(CC) $(CFLAGS) -c test.c -o $(BINARY_DIR)/test.o
	$(CC) $(CFLAGS) $(BINARY_DIR)/test.o -o $(BINARY_DIR)/test.out $(LIBS)
	./$(BINARY_DIR)/test.out


update_lib:
	mkdir -p $(BINARY_DIR)/
	cp ../lib/libblasfeo.a ./$(BINARY_DIR)
	cp ../lib/libblasfeo_ref.a ./$(BINARY_DIR)

run: test.o

update: run
full: update_lib run

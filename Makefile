# 定义源代码目录
SRC_DIR := core

# 递归查找所有的 cpp 和 h 文件
CPP_FILES := $(shell find $(SRC_DIR) -name "*.cc" -o)
H_FILES := $(shell find $(SRC_DIR) -name "*.h")
SOURCES := $(CPP_FILES) $(H_FILES)

# 该文件所在文件夹
CUR_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

format:
	@echo "Formatting all source files..."
	@for src in $(SOURCES) ; do \
		echo "Formatting $$src..." ; \
		clang-format -i "$$src" ; \
	done
	@echo "Done"

check-style:
	@echo "Checking coding style..."
	@for src in $(SOURCES) ; do \
		var=`clang-format "$$src" | diff "$$src" - | wc -l` ; \
		if [ $$var -ne 0 ] ; then \
			echo "$$src does not respect the coding style (diff: $$var lines)" ; \
			exit 1 ; \
		fi ; \
	done
	@echo "Style check passed"

tidy:
	@echo "Running clang-tidy on all source files..."
	@for src in $(CPP_FILES) ; do \
		echo "Running tidy on $$src..." ; \
		clang-tidy "$$src" ; \
	done
	@echo "Done"

# 添加一个清理临时文件的目标
# clean:
# 	@echo "Cleaning temporary files..."
# 	@find $(SRC_DIR) -name "*.o" -delete
# 	@find $(SRC_DIR) -name "*.d" -delete
# 	@echo "Done"

.PHONY: format check-style tidy # clean

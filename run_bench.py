from perfboard import module_classes, TestModule

if __name__ == '__main__':
    all_tests = []
    for mod in module_classes:
        if hasattr(mod, 'TESTS'):
            all_tests += [TestModule(mod(*args)) for args in mod.TESTS]

    for test in all_tests:
        test.run('torch', repeat=1)

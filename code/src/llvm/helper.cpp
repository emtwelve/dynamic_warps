int test (bool x, int y, int z) {
    printf("%d,%d\n",y,z);
    int q;
    if (x) {
        q = y + z;
    } else {
        q = z - y;
    }
    return q;
}

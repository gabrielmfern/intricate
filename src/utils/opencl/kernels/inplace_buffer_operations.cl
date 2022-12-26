kernel void clip_min_max(
    global float *self,
    
    float _min,
    float _max,
    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = max(min((float)self[index], _min), _max);
}

kernel void scale(
    global float *self,
    
    float scaler,
    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = (float)self[index] * scaler;
}

kernel void sqrt(
    global float *buf,
    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    buf[index] = sqrt(buf[index]);
}

kernel void inverse_sqrt(
    global float *buf,
    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    buf[index] = rsqrt(buf[index]);
}

kernel void shift(
    global float *buf,

    float num,
    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    buf[index] = buf[index] + num;
}

kernel void add(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] + other[index];
}

kernel void subtract(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] - other[index];
}

kernel void multiply(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] * other[index];
}

kernel void divide(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] / other[index];
}
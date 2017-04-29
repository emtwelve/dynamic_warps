int main(int argc, char** argv)
{
    parse_in_csv(input_array, filename);

    createGraph(input_array, result_array, N);

    fill_out_csv(result_array, filename);
}


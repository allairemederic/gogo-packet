module packet_parser #(
    parameter int IP_ADDRESS_SIZE = 32
)(
    input logic clk,
    input logic rst_n,

    // Inputs
    input logic[7:0] data_in,
    input logic valid_in,
    output logic ready_in,

    // Outputs
    output logic[7:0] data_out,
    output logic valid_out,
    input logic ready_out

);


endmodule: packet_parser
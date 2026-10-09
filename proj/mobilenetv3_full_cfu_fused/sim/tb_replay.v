`timescale 1ns/1ps
// Replays <dir>/cmds.txt (custom-instruction log from the real C driver) into the RTL and writes the
// CFU response of EVERY instruction to <dir>/resp.txt.  Used for the two-pass block tests.
module tb_replay;
    reg clk = 0, reset = 1;
    always #5 clk = ~clk;
    reg cmd_valid = 0; reg [9:0] fid = 0; reg [31:0] in0 = 0, in1 = 0;
    wire cmd_ready, rsp_valid; reg rsp_ready = 1; wire [31:0] rsp;
    Cfu dut (.cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_payload_function_id(fid),
             .cmd_payload_inputs_0(in0), .cmd_payload_inputs_1(in1),
             .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_payload_outputs_0(rsp),
             .reset(reset), .clk(clk));
    integer fd, fo, r, ncmd = 0;
    reg [31:0] op, a, b, resp;
    reg [1023:0] dir; reg [1023:0] fname;
    task do_cmd(input [6:0] f7, input [31:0] x, input [31:0] y);
        begin
            @(negedge clk); fid = {f7, 3'b000}; in0 = x; in1 = y; cmd_valid = 1;
            @(posedge clk); while (!cmd_ready) @(posedge clk);
            #1 cmd_valid = 0;
            @(posedge clk); while (!rsp_valid) @(posedge clk);
            resp = rsp;
            repeat (2) @(posedge clk);
        end
    endtask
    initial begin
        if (!$value$plusargs("dir=%s", dir)) dir = "build";
        repeat (8) @(posedge clk); reset = 0; repeat (4) @(posedge clk);
        $sformat(fname, "%0s/cmds.txt", dir); fd = $fopen(fname, "r");
        $sformat(fname, "%0s/resp.txt", dir); fo = $fopen(fname, "w");
        if (fd == 0) begin $display("cannot open cmds"); $finish; end
        while (!$feof(fd)) begin
            r = $fscanf(fd, "%h %h %h\n", op, a, b);
            if (r == 3) begin do_cmd(op[6:0], a, b); $fdisplay(fo, "%08x", resp); ncmd = ncmd + 1; end
        end
        $fclose(fd); $fclose(fo);
        $display("[tb] replayed %0d instructions", ncmd);
        $finish;
    end
    initial begin #4000000000; $display("TIMEOUT"); $finish; end
endmodule

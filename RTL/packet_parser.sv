module packet_parser #(
    parameter int MAX_FRAME_BYTES = 32768
)(
    input logic clk,
    input logic rst_n,

    // Inputs
    input  logic[7:0] data_in,
    input  logic      valid_in,
    output logic      ready_in,
    input  logic      last_in,

    // Outputs
    output logic[7:0] data_out,
    output logic      valid_out,
    input  logic      ready_out, 
    output logic      last_out,

);

    localparam int COUNT_WIDTH = $clog2(MAX_FRAME_BYTES + 1);

    typedef enum logic [2:0] {
        ASSEMBLE_PACKET,
        PARSE_ETHERNET,
        PARSE_IP,
        PARSE_TCP,
        PARSE_UDP,
        SEND_RESULT
    } state_t;

    typedef enum logic [1:0] { 
        VALID, 
        INVALID, 
        UNSUPPORTED
    } status_t;

    state_t state;
    status_t status;

    // Pour l'assemblage de la trame
    logic [7:0] packet_buffer [0:MAX_FRAME_BYTES-1];
    logic [COUNT_WIDTH-1:0] byte_count;
    logic overflow;

    // Champs Ethernet
    logic [15:0] ether_type;

    // Champs IPv4
    logic [3:0]  ip_version;
    logic [7:0]  ip_header_length;
    logic [15:0] ip_length;
    logic [7:0]  protocol;
    logic [31:0] src_ip;
    logic [31:0] dst_ip;

    // Champs TCP ou UDP
    logic [15:0] src_port;
    logic [15:0] dst_port;
    logic [15:0] transport_length;
    logic [15:0] payload_length;
    logic [7:0]  tcp_header_length;
    logic [15:0] udp_length;

    // Valeurs lues directement dans le buffer
    logic [7:0]  ip_header_bytes;
    logic [15:0] ip_total_bytes;
    logic [15:0] ip_fragment;
    logic [7:0]  tcp_header_bytes;
    logic [15:0] udp_total_bytes;

    // Position du début de TCP ou UDP
    logic [6:0] transport_start;

    // Position de l'octet à envoyer : de 0 à 19
    logic [4:0] result_index;

    // IPv4 commence après les 14 octets Ethernet
    assign ip_header_bytes = { 2'b00, packet_buffer[14][3:0], 2'b00 };
    assign ip_total_bytes  = { packet_buffer[16], packet_buffer[17] };
    assign ip_fragment     = { packet_buffer[20], packet_buffer[21] };

    // Utiliser la longueur IPv4 enregistrée dans PARSE_IP
    assign transport_start = 7'd14 + ip_header_length[6:0];

    // La longueur TCP est exprimée en mots de 4 octets
    assign tcp_header_bytes = { 2'b00, packet_buffer[transport_start + 12][7:4], 2'b00 };
    assign udp_total_bytes = { packet_buffer[transport_start + 4], packet_buffer[transport_start + 5] };

    // Accepter un octet seulement quand on assemble
    assign ready_in = (state == ASSEMBLE_PACKET);

    // Présenter les octets pendant l'envoi du résultat
    assign valid_out = (state == SEND_RESULT);

    // Signaler le dernier octet
    assign last_out = valid_out && (result_index == 5'd19);

    // FSM
    always_ff @(posedge clock or negedge rst_n ) begin

        if (!rst_n) begin
            state             <= ASSEMBLE_PACKET;
            byte_count        <= '0;
            ether_type        <= '0;
            overflow          <= 1'b0;
            status            <= VALID;
            ip_version        <= '0;
            ip_header_length  <= '0;
            ip_length         <= '0;
            protocol          <= '0;
            src_ip            <= '0;
            dst_ip            <= '0;
            src_port          <= '0;
            dst_port          <= '0;
            transport_length  <= '0;
            payload_length    <= '0;
            tcp_header_length <= '0;
            udp_length        <= '0;   
            result_index      <= '0;     
        end

        else begin
            case(state)

                // On assemble la trame complete avant de la traiter
                ASSEMBLE_PACKET: begin

                    // Accepte l'octet entrant seulement quand le module est pret
                    if (valid_in && ready_in) begin

                        if (byte_count < MAX_FRAME_BYTES) begin
                            packet_buffer[byte_count] <= data_in;
                            byte_count <= byte_count + 1'b1;
                        end

                        else begin
                            overflow <= 1'b1;
                        end

                        // Dernier octet recu, on peut passer au traitement
                        if (last_in) begin
                            state <= PARSE_ETHERNET;
                        end
                    end
                end

                // Analyse de l'en-tete Ethernet
                PARSE_ETHERNET: begin

                    // Trame trop longue ou en-tête Ethernet incomplet
                    if (overflow || byte_count < 14) begin
                        status <= INVALID;
                    end

                    else begin

                        // Lire l'EtherType aux octets 12 et 13 (sans VLAN)
                        ether_type <= {packet_buffer[12], packet_buffer[13]};

                        // Comparer les octets
                        if ({packet_buffer[12], packet_buffer[13]} == 16'h0800) begin
                            status <= VALID;
                            state  <= PARSE_IP;
                        end
                        else begin
                            status <= UNSUPPORTED;
                            result_index <= '0;
                            state  <= SEND_RESULT;
                        end
                    end
                end

                // Analyse de l'en-tete IPv4
                PARSE_IP: begin

                    // Vérifier la présence des 20 octets IPv4 minimum
                    if (byte_count < 34) begin
                        status <= INVALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    // Vérifier la version et les longueurs annoncées
                    else if (packet_buffer[14][7:4] != 4 || ip_header_bytes < 20 ||  ip_total_bytes < ip_header_bytes) begin
                        status <= INVALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    // Vérifier que tout le paquet IP a été recu
                    // Élargir l'addition pour éviter un overflow
                    else if (byte_count < (17'd14 + {1'b0, ip_total_bytes})) begin
                        status <= INVALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    // Ne pas traiter les fragments IPv4.
                    else if ((ip_fragment & 16'h3FFF) != 0) begin
                        status <= UNSUPPORTED;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    else begin
                        // Enregistrer les champs IPv4
                        ip_version       <= packet_buffer[14][7:4];
                        ip_header_length <= ip_header_bytes;
                        ip_length        <= ip_total_bytes;
                        protocol         <= packet_buffer[23];

                        src_ip <= {packet_buffer[26], packet_buffer[27], packet_buffer[28], packet_buffer[29]};
                        dst_ip <= {packet_buffer[30], packet_buffer[31], packet_buffer[32], packet_buffer[33]};

                        // Nombre d'octets après l'en-tête IPv4
                        transport_length <= ip_total_bytes - ip_header_bytes;

                        // Choisir l'en-tête suivant
                        case (packet_buffer[23])

                            // TCP
                            8'd6: begin
                                state <= PARSE_TCP;
                            end

                            // UDP
                            8'd17: begin
                                state <= PARSE_UDP;
                            end

                            default: begin
                                status <= UNSUPPORTED;
                                result_index <= '0;
                                state  <= SEND_RESULT;
                            end
                        endcase
                    end
                end
                

                // Analyse de l'en-tete TCP
                PARSE_TCP: begin

                    // TCP doit contenir au moins 20 octets d'en-tête
                    if (transport_length < 20) begin
                        status <= INVALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    // Vérifier la longueur de l'en-tête TCP
                    else if (tcp_header_bytes < 20 || tcp_header_bytes > transport_length) begin
                        status <= INVALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    else begin
                        // Lire les ports source et destination
                        src_port <= {packet_buffer[transport_start], packet_buffer[transport_start + 1]};
                        dst_port <= {packet_buffer[transport_start + 2], packet_buffer[transport_start + 3]};

                        tcp_header_length <= tcp_header_bytes;

                        // Retirer l'en-tête TCP pour obtenir le payload
                        payload_length <= transport_length - tcp_header_bytes;
                        status <= VALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end
                    
                end

                // Analyse de l'en-tete UDP
                PARSE_UDP: begin

                    // UDP doit contenir ses 8 octets d'en-tête
                    if (transport_length < 8) begin
                        status <= INVALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    // Vérifier la longueur annoncée par UDP
                    else if ( udp_total_bytes < 8 || udp_total_bytes > transport_length) begin
                        status <= INVALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end

                    else begin
                        // Lire les ports source et destination
                        src_port <= {packet_buffer[transport_start], packet_buffer[transport_start + 1]};
                        dst_port <= {packet_buffer[transport_start + 2], packet_buffer[transport_start + 3]};

                        udp_length <= udp_total_bytes;

                        // Retirer les 8 octets de l'en-tête UDP
                        payload_length <= udp_total_bytes - 16'd8;

                        status <= VALID;
                        result_index <= '0;
                        state  <= SEND_RESULT;
                    end
                end

                // Envoyer les 20 octets du résultat
                SEND_RESULT: begin
                    // Avancer seulement lorsque l'octet est accepté
                    if (valid_out && ready_out) begin

                        if (result_index == 5'd19) begin

                            // Le résultat complet a été envoyé
                            byte_count   <= '0;
                            overflow     <= 1'b0;
                            result_index <= '0;
                            status       <= VALID;
                            state <= ASSEMBLE_PACKET;
                        end

                        else begin
                            result_index <= result_index + 1'b1;
                        end
                    end
                end

            endcase
        end
    end


    // Pour piloter la sortie
    always_comb begin
        data_out = 8'd0;

        if (state == SEND_RESULT) begin

            // Toujours envoyer le statut en premier
            if (result_index == 0) begin
                data_out = {6'b0, status};
            end

            // Pour une trame rejetée, les autres octets restent à zéro
            else if (status == VALID) begin
                case (result_index)

                    5'd1:  data_out = {4'b0, ip_version};
                    5'd2:  data_out = protocol;
                    5'd3:  data_out = 8'd0;

                    // Adresse IP source
                    5'd4:  data_out = src_ip[31:24];
                    5'd5:  data_out = src_ip[23:16];
                    5'd6:  data_out = src_ip[15:8];
                    5'd7:  data_out = src_ip[7:0];

                    // Adresse IP destination
                    5'd8:  data_out = dst_ip[31:24];
                    5'd9:  data_out = dst_ip[23:16];
                    5'd10: data_out = dst_ip[15:8];
                    5'd11: data_out = dst_ip[7:0];

                    // Ports source et destination
                    5'd12: data_out = src_port[15:8];
                    5'd13: data_out = src_port[7:0];
                    5'd14: data_out = dst_port[15:8];
                    5'd15: data_out = dst_port[7:0];

                    // Longueur du paquet IP
                    5'd16: data_out = ip_length[15:8];
                    5'd17: data_out = ip_length[7:0];

                    // Longueur du payload
                    5'd18: data_out = payload_length[15:8];
                    5'd19: data_out = payload_length[7:0];

                    default: data_out = 8'd0;

                endcase
            end
        end
    end


endmodule: packet_parser
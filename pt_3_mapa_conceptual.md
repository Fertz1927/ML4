graph TD
    %% Definición de Estilos
    classDef nodoPrincipal fill:#2B3A67,stroke:#ffffff,stroke-width:3px,color:#ffffff,font-weight:bold;
    classDef nodoSecundario fill:#496A81,stroke:#ffffff,stroke-width:1px,color:#ffffff;
    classDef nodoDetalle fill:#6699A1,stroke:#ffffff,stroke-width:1px,color:#ffffff,font-style:italic;
    classDef nodoAplicacion fill:#D4A373,stroke:#333333,stroke-width:2px,color:#333333,font-weight:bold;

    %% =========================================================
    %% NODO 2: BASES ADAPTATIVAS VS RÍGIDAS (Origen del flujo)
    %% =========================================================
    N2[2. BASES DE REPRESENTACIÓN]:::nodoPrincipal
    
    N2 -- Independientes de los datos --> N2_Rig[Bases Rígidas]:::nodoSecundario
    N2 -- Dependientes de los datos --> N2_Ada[Bases Adaptativas]:::nodoSecundario
    
    N2_Rig --> N2_R_Four[Fourier / DCT]:::nodoDetalle
    N2_Rig --> N2_R_Wav[Wavelets]:::nodoDetalle
    
    N2_Ada --> N2_A_PCA[SVD / PCA / Karhunen-Loève]:::nodoDetalle

    %% =========================================================
    %% NODO 3: LOCALIZACIÓN (Deriva de Bases Rígidas)
    %% =========================================================
    N3[3. LOCALIZACIÓN ESPACIO/TIEMPO-FRECUENCIA]:::nodoPrincipal
    
    N2_R_Four -- Analiza en --> N3_Frec[Frecuencia Pura<br>Sin localización temporal]:::nodoSecundario
    N2_R_Wav -- Analiza mediante --> N3_Multi[Multirresolución y Soporte Compacto<br>Localiza tiempo y frecuencia]:::nodoSecundario
    
    N3_Frec --- N3
    N3_Multi --- N3

    %% =========================================================
    %% NODO 1: REDUCCIÓN DE DIMENSIONALIDAD (Deriva de Bases Adaptativas)
    %% =========================================================
    N1[1. REDUCCIÓN DE DIMENSIONALIDAD]:::nodoPrincipal
    
    N2_A_PCA -- Ejecuta mediante --> N1_Lin[Métodos Lineales<br>Proyección Ortogonal de Subespacios]:::nodoSecundario
    N1_Lin --- N1
    N1 -- Contrasta con --> N1_NLin[Métodos No Lineales<br>Manifolds, Kernel PCA, Autoencoders]:::nodoSecundario

    %% =========================================================
    %% NODO 4: CRITERIOS DE TRUNCAMIENTO Y SPARSITY
    %% =========================================================
    N4[4. CRITERIOS DE TRUNCAMIENTO Y SPARSITY]:::nodoPrincipal
    
    N1 -.->|Requiere elegir cuántos componentes retener| N4
    N3 -.->|Requiere filtrar coeficientes irrelevantes| N4
    
    N4 -- Para SVD/PCA --> N4_Var[Varianza Explicada]:::nodoSecundario
    N4 -- Para Wavelets/Fourier --> N4_Ene[Umbralizado de Energía<br>Hard/Soft Thresholding]:::nodoSecundario
    N4 -- Regularización ML --> N4_Norm[Normas Matemáticas:<br>L1 promueve Sparsity vs L2 promueve Suavidad]:::nodoSecundario

    %% =========================================================
    %% NODO 5: APLICACIONES EN ML Y FÍSICA
    %% =========================================================
    N5[5. APLICACIONES EN ML Y FÍSICA]:::nodoPrincipal
    
    N4_Var -->|Habilita| N5
    N4_Ene -->|Habilita| N5
    N4_Norm -->|Habilita| N5
    
    N5 --> App1[Denoising <br>Limpieza de Señales/Imágenes]:::nodoAplicacion
    N5 --> App2[Extracción de Características <br>Inputs para Modelos de ML]:::nodoAplicacion
    N5 --> App3[Compresión de Datos <br>JPEG/JPEG2000, Audio]:::nodoAplicacion
    N5 --> App4[Descomposición de Modos Propios <br>Mecánica de Fluidos: POD / DMD]:::nodoAplicacion

# multistage
FROM golang:1.25-bookworm AS build

WORKDIR /app

RUN useradd -u 1001 nonroot

COPY go.mod go.sum ./

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go mod download

COPY . .

#compiles your Go source code into an executable binary file
RUN go build \
    -ldflags="-linkmode external -extldflags -static" \
    -tags netgo \
    -o api-golang
#-o api-golang: The -o flag stands for output. It tells the compiler to name the resulting executable api-golang instead of using the default name (which is usually the name of your folder or main).

###
FROM  scratch

ENV GIN_MODE=release

COPY --from=build /etc/passwd /etc/passwd

COPY --from=build /app/api-golang api-golang

USER nonroot

EXPOSE 8080

CMD ["./api-golang"]